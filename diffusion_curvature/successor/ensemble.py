"""Vmapped multi-seed FB training: S seeds in one batched forward pass.

Seed ensembles are required for stable curvature estimates (single-run FB is
materially nondeterministic), but serial training makes them S x expensive.
``EnsembleFBTrainer`` stacks S independently-initialized (F, B) nets with
``torch.func.stack_module_state`` and trains them simultaneously under vmap —
the same TD-backup contrastive Bellman-gap loss as ``FBTrainer``, per seed,
at roughly the wall cost of one net.

Each seed keeps its own init, its own target networks, and its own data
order; gradients never mix across seeds (the losses are summed, but the
parameters are disjoint slices of the stacked tensors).
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import torch
from torch.func import functional_call, stack_module_state, vmap

from .fb_modules import BackwardMap, ForwardMap
from .train import _collect_transitions


class EnsembleFBTrainer:
    """Train S seeds of (F, B) simultaneously via torch.func.vmap.

    Mirrors ``FBTrainer``'s hyperparameters; ``seeds`` replaces ``seed``.
    After ``fit``, use ``raw_kernels(X)`` for the (S, n, n) symmetrized
    unclipped score matrices min(F1 B^T, F2 B^T).
    """

    def __init__(
        self,
        obs_dim: int,
        seeds: Sequence[int] = (7, 8, 9, 10),
        z_dim: int = 64,
        hidden_dim: int = 256,
        gamma: float = 0.98,
        tau_polyak: float = 0.01,
        ortho_coef: float = 1.0,
        lr: float = 1e-4,
        lr_min: float = 1e-6,
        cosine_lr: bool = True,
        n_epochs: int = 600,
        batch_size: int = 1024,
        device: str | torch.device = "cuda:0",
    ):
        self.seeds = tuple(seeds)
        self.gamma = gamma
        self.tau_polyak = tau_polyak
        self.ortho_coef = ortho_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        F_models, B_models = [], []
        for s in self.seeds:
            torch.manual_seed(s)
            F_models.append(ForwardMap(obs_dim, z_dim=z_dim,
                                       hidden_dim=hidden_dim).to(self.device))
            B_models.append(BackwardMap(obs_dim, z_dim=z_dim,
                                        hidden_dim=hidden_dim).to(self.device))
        self.pF, self.bF = stack_module_state(F_models)
        self.pB, self.bB = stack_module_state(B_models)
        for p in (*self.pF.values(), *self.pB.values()):
            p.requires_grad_(True)
        # Polyak targets: plain stacked tensors, no grad.
        self.tpF = {k: v.detach().clone() for k, v in self.pF.items()}
        self.tpB = {k: v.detach().clone() for k, v in self.pB.items()}

        # 'meta' skeletons for functional_call
        self._baseF = copy.deepcopy(F_models[0]).to("meta")
        self._baseB = copy.deepcopy(B_models[0]).to("meta")

        self.optimizer = torch.optim.Adam(
            [*self.pF.values(), *self.pB.values()], lr=lr)
        self.scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs, eta_min=lr_min)
            if cosine_lr else None
        )

    # -- vmapped loss ---------------------------------------------------------

    def _seed_loss(self, pF, pB, tpF, tpB, obs, nxt):
        tF1, tF2 = functional_call(self._baseF, (tpF, self.bF), (nxt,))
        tB = functional_call(self._baseB, (tpB, self.bB), (nxt,))
        tM = torch.minimum(tF1 @ tB.T, tF2 @ tB.T).detach()

        F1, F2 = functional_call(self._baseF, (pF, self.bF), (obs,))
        B = functional_call(self._baseB, (pB, self.bB), (nxt,))
        M1, M2 = F1 @ B.T, F2 @ B.T

        off = ~torch.eye(M1.size(0), device=M1.device, dtype=torch.bool)
        loss_offdiag = 0.5 * sum(
            (M - self.gamma * tM)[off].pow(2).mean() for M in (M1, M2))
        loss_diag = -sum(M.diagonal().mean() for M in (M1, M2))
        Cov = B @ B.T
        orth = Cov[off].pow(2).mean() - 2 * Cov.diagonal().mean()
        return loss_offdiag + loss_diag + self.ortho_coef * orth

    def fit(self, trajectories) -> "EnsembleFBTrainer":
        obs_all, nxt_all = _collect_transitions(trajectories)
        n = obs_all.shape[0]
        if n < 4:
            raise ValueError(f"Need >= 4 transitions, got {n}.")
        obs_t = torch.as_tensor(obs_all, dtype=torch.float32, device=self.device)
        nxt_t = torch.as_tensor(nxt_all, dtype=torch.float32, device=self.device)
        rngs = [np.random.default_rng(s) for s in self.seeds]
        S = len(self.seeds)
        loss_fn = vmap(self._seed_loss)

        for epoch in range(self.n_epochs):
            perms = np.stack([r.permutation(n) for r in rngs])  # (S, n)
            for start in range(0, n, self.batch_size):
                idx = perms[:, start:start + self.batch_size]
                if idx.shape[1] < 4:
                    continue
                idx_t = torch.as_tensor(idx, device=self.device)
                obs = obs_t[idx_t]          # (S, B, d)
                nxt = nxt_t[idx_t]

                losses = loss_fn(self.pF, self.pB, self.tpF, self.tpB, obs, nxt)
                self.optimizer.zero_grad(set_to_none=True)
                losses.sum().backward()
                self.optimizer.step()

                with torch.no_grad():
                    for k in self.tpF:
                        self.tpF[k].mul_(1 - self.tau_polyak).add_(
                            self.pF[k].detach(), alpha=self.tau_polyak)
                    for k in self.tpB:
                        self.tpB[k].mul_(1 - self.tau_polyak).add_(
                            self.pB[k].detach(), alpha=self.tau_polyak)

            if self.scheduler is not None:
                self.scheduler.step()
        return self

    # -- inference ------------------------------------------------------------

    @torch.no_grad()
    def raw_kernels(self, X: np.ndarray) -> np.ndarray:
        """(S, n, n) symmetrized unclipped kernels, one per seed."""
        o = torch.as_tensor(X, dtype=torch.float32, device=self.device)

        def _one(pF, pB):
            F1, F2 = functional_call(self._baseF, (pF, self.bF), (o,))
            B = functional_call(self._baseB, (pB, self.bB), (o,))
            return torch.minimum(F1 @ B.T, F2 @ B.T)

        K = vmap(_one)(self.pF, self.pB).cpu().numpy()
        return 0.5 * (K + K.transpose(0, 2, 1))
