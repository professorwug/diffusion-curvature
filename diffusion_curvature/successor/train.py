"""Co-train F and B successor-measure networks from trajectories.

Ported from reason_reckon's ``torus-baseline/fb_torus.py``, which is the
codepath that produced the ground-truth curvature signal analysis on the
torus — the direct validation of FB as a scalar-curvature estimator on
manifolds. Same architectures, same TD-backup contrastive Bellman-gap loss,
same orthogonality regularizer on B, same Polyak-averaged target networks.

Training objective::

    L = loss_offdiag + loss_diag + ortho_coef * orth(B)

for each batch of consecutive (obs, next_obs) pairs::

    F1, F2 = F_net(obs);  B = B_net(next_obs)
    Mk = Fk @ B.T                               for k in {1, 2}
    tF1, tF2 = F_target(next_obs);  tB = B_target(next_obs)
    tM = min(tF1 @ tB.T, tF2 @ tB.T)            # stop-grad
    loss_offdiag = 0.5 * mean_k (Mk - gamma * tM)[off]^2
    loss_diag    = -sum_k mean(Mk.diag())
    orth(B)      = mean((B @ B.T)[off]^2) - 2 * mean((B @ B.T).diag())

Training is full-epoch mini-batched (shuffle transitions each epoch).
Optional cosine LR schedule matches fb_torus.
"""

from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import torch
from torch import nn

from .fb_modules import BackwardMap, ForwardMap


def _freeze(net: nn.Module) -> nn.Module:
    for p in net.parameters():
        p.requires_grad = False
    return net


def _soft_update(net: nn.Module, target: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for p, tp in zip(net.parameters(), target.parameters()):
            tp.data.mul_(1 - tau).add_(p.data, alpha=tau)


def _collect_transitions(
    trajectories: Sequence[np.ndarray] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten trajectories into consecutive (obs, next_obs) arrays."""
    if isinstance(trajectories, np.ndarray) and trajectories.ndim == 3:
        trajs = [trajectories[i] for i in range(trajectories.shape[0])]
    else:
        trajs = list(trajectories)
    obs_list, nxt_list = [], []
    for traj in trajs:
        if traj.shape[0] < 2:
            continue
        obs_list.append(traj[:-1])
        nxt_list.append(traj[1:])
    if not obs_list:
        raise ValueError("No valid transitions (need trajectories with length ≥ 2).")
    return np.concatenate(obs_list, axis=0), np.concatenate(nxt_list, axis=0)


def fb_loss(
    F_net: ForwardMap,
    B_net: BackwardMap,
    F_target: ForwardMap,
    B_target: BackwardMap,
    obs: torch.Tensor,
    next_obs: torch.Tensor,
    gamma: float,
    ortho_coef: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Contrastive Bellman-gap loss + orthogonality regularizer on B."""
    with torch.no_grad():
        tF1, tF2 = F_target(next_obs)
        tB = B_target(next_obs)
        tM = torch.minimum(tF1 @ tB.T, tF2 @ tB.T)

    F1, F2 = F_net(obs)
    B = B_net(next_obs)
    M1 = F1 @ B.T
    M2 = F2 @ B.T

    I_mat = torch.eye(M1.size(0), device=M1.device)
    off = ~I_mat.bool()

    loss_offdiag = 0.5 * sum(
        (M - gamma * tM)[off].pow(2).mean() for M in (M1, M2)
    )
    loss_diag = -sum(M.diag().mean() for M in (M1, M2))

    Cov = B @ B.T
    orth = Cov[off].pow(2).mean() - 2 * Cov.diag().mean()

    loss = loss_offdiag + loss_diag + ortho_coef * orth
    metrics = {
        "fb_loss": float(loss.item()),
        "loss_offdiag": float(loss_offdiag.item()),
        "loss_diag": float(loss_diag.item()),
        "orth": float(orth.item()),
        "B_norm": float(torch.norm(B, dim=-1).mean().item()),
    }
    return loss, metrics


class FBTrainer:
    """Own and co-train F, B successor-measure networks from trajectories.

    Defaults follow reason_reckon's ``torus-baseline/fb_torus.py`` — the codepath
    calibrated for scalar-curvature recovery on 2-manifolds. For high-dim LLM-
    embedding contexts (cf. ``ee_bonus.py``), set ``z_dim=16, hidden_dim=128,
    gamma=0.8``.

    Parameters
    ----------
    obs_dim : int
        Ambient observation dimension.
    z_dim : int, default 2
        Successor-measure latent dimension. 2 recovers eigenmap-style
        coordinates on a 2-manifold; raise for higher-d data.
    hidden_dim : int, default 256
        MLP width for both F and B.
    gamma : float, default 0.5
        Successor discount for the TD backup.
    tau_polyak : float, default 0.01
        Soft-update rate for target networks.
    ortho_coef : float, default 1.0
    lr : float, default 1e-4
        Peak LR (also fixed LR when ``cosine_lr=False``).
    lr_min : float, default 1e-6
        Final LR under cosine schedule.
    cosine_lr : bool, default False
    n_epochs : int, default 200
        Full passes over the transition set.
    batch_size : int, default 512
    device : str | torch.device, default 'cpu'
    seed : int, default 42
    """

    def __init__(
        self,
        obs_dim: int,
        z_dim: int = 2,
        hidden_dim: int = 256,
        gamma: float = 0.5,
        tau_polyak: float = 0.01,
        ortho_coef: float = 1.0,
        lr: float = 1e-4,
        lr_min: float = 1e-6,
        cosine_lr: bool = False,
        n_epochs: int = 200,
        batch_size: int = 512,
        device: str | torch.device = "cpu",
        seed: int = 42,
    ):
        torch.manual_seed(seed)
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.tau_polyak = tau_polyak
        self.ortho_coef = ortho_coef
        self.lr = lr
        self.lr_min = lr_min
        self.cosine_lr = cosine_lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.seed = seed

        self.F_net = ForwardMap(obs_dim, z_dim=z_dim, hidden_dim=hidden_dim).to(self.device)
        self.B_net = BackwardMap(obs_dim, z_dim=z_dim, hidden_dim=hidden_dim).to(self.device)
        self.F_target = _freeze(copy.deepcopy(self.F_net))
        self.B_target = _freeze(copy.deepcopy(self.B_net))

        self.optimizer = torch.optim.Adam(
            [*self.F_net.parameters(), *self.B_net.parameters()],
            lr=self.lr,
        )
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        if cosine_lr:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=n_epochs, eta_min=self.lr_min,
            )
        self.history: list[dict[str, float]] = []

    def fit(
        self,
        trajectories: Sequence[np.ndarray] | np.ndarray,
        log_every: int | None = 50,
    ) -> "FBTrainer":
        obs_all, nxt_all = _collect_transitions(trajectories)
        n = obs_all.shape[0]
        if n < 4:
            raise ValueError(
                f"Need at least 4 transitions for off-diagonal loss; got {n}."
            )

        rng = np.random.default_rng(self.seed)
        self.F_net.train()
        self.B_net.train()

        for epoch in range(self.n_epochs):
            perm = rng.permutation(n)
            epoch_metrics: dict[str, float] = {}
            n_batches = 0
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                if len(idx) < 4:
                    continue
                obs = torch.as_tensor(obs_all[idx], dtype=torch.float32, device=self.device)
                nxt = torch.as_tensor(nxt_all[idx], dtype=torch.float32, device=self.device)

                loss, metrics = fb_loss(
                    self.F_net, self.B_net,
                    self.F_target, self.B_target,
                    obs, nxt,
                    self.gamma, self.ortho_coef,
                )
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                _soft_update(self.F_net, self.F_target, self.tau_polyak)
                _soft_update(self.B_net, self.B_target, self.tau_polyak)

                for k, v in metrics.items():
                    epoch_metrics[k] = epoch_metrics.get(k, 0.0) + v
                n_batches += 1

            if self.scheduler is not None:
                self.scheduler.step()

            if log_every and (epoch % log_every == 0 or epoch == self.n_epochs - 1):
                entry = {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}
                entry["epoch"] = float(epoch)
                entry["lr"] = self.optimizer.param_groups[0]["lr"]
                self.history.append(entry)
        return self
