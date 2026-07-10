"""Variational (lower-bound) estimators of the successor-MI curvature field.

The successor entropy H(S+|s) upper-bounds the empowerment I(A;S+|s); these
estimators approach the same field from below, learning it from trajectory
PAIRS alone — no transition counts, no matrix inversion, optionally no
coordinates.

V1 — `VMI`: InfoNCE-trained critic f(s,s') = F(s)·B(s') + b(s') on pairs
(s_t, s_{t+k}), k ~ Geom(1-gamma) (the resolvent horizon). At optimum
e^f ∝ M_gamma(s,·)/rho(·). Readouts per anchor state:
  mi_bound  — Donsker–Varadhan bound on KL(M(s,·) || unif) = log n - H(s),
              using the anchor's HELD-OUT future visits as samples from M.
  mi_plugin — plug-in KL of the critic's full-corpus softmax row.
  H_plugin  — Shannon entropy of that softmax row (plug-in sent).

Curvature orientation: +mi (equivalently -H).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def harvest_pairs(traj: np.ndarray, gamma: float, lags_per_step: int,
                  rng: np.random.Generator,
                  min_lag: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Sample (anchor_state, future_state) pairs from walks with
    k ~ Geom(1-gamma) lags (k >= min_lag), truncated at walk ends.
    min_lag > 1 decorrelates pairs from temporally correlated observation
    noise (skip lags inside the noise correlation time).

    traj: (nt, T+1) int array of state indices.
    Returns (anchors, positives) as int arrays.
    """
    nt, T1 = traj.shape
    T = T1 - 1
    t_idx = np.tile(np.arange(T), nt)
    w_idx = np.repeat(np.arange(nt), T)
    a_list, p_list = [], []
    for _ in range(lags_per_step):
        k = rng.geometric(p=1.0 - gamma, size=t_idx.size)
        if min_lag > 1:
            k = np.maximum(k, min_lag)
        ok = t_idx + k <= T
        a_list.append(traj[w_idx[ok], t_idx[ok]])
        p_list.append(traj[w_idx[ok], t_idx[ok] + k[ok]])
    return np.concatenate(a_list), np.concatenate(p_list)


def noise_structure_cv(X: np.ndarray, traj: np.ndarray,
                       n_probes: int = 256, m_ball: int = 200,
                       rng=None) -> float:
    """Coefficient of variation of the LOCAL noise floor across the manifold.

    Local nugget: for probe transitions grouped by region, intercept of
    MSD(k) at k=0 estimates 2*sigma_obs^2(x). Structured (spatially varying)
    noise has high CV; isotropic noise (any magnitude) has low CV. Used to
    auto-switch the jitter rule: structured noise wants uncorrected
    over-smoothing; unstructured noise wants the nugget subtraction.
    """
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(rng)
    nt, T1 = traj.shape
    w = rng.integers(0, nt, 20000)
    t = rng.integers(0, T1 - 2, 20000)
    x0 = X[traj[w, t]]
    d1 = ((X[traj[w, t + 1]] - x0)**2).sum(1)
    d2 = ((X[traj[w, t + 2]] - x0)**2).sum(1)
    probes = X[traj[rng.integers(0, nt, n_probes),
                    rng.integers(0, T1, n_probes)]]
    tree = cKDTree(x0)
    sig2 = []
    for q in range(n_probes):
        _, idx = tree.query(probes[q], k=m_ball)
        loc = max(0.0, float(np.mean(d1[idx])
                             - (np.mean(d2[idx]) - np.mean(d1[idx])))) / 2
        sig2.append(loc)
    sig2 = np.asarray(sig2)
    mu = float(np.mean(sig2))
    if mu <= 1e-12:
        return 0.0
    return float(np.std(sig2) / mu)


class _TabularCritic(nn.Module):
    def __init__(self, n_states: int, z_dim: int):
        super().__init__()
        self.F = nn.Embedding(n_states, z_dim)
        self.B = nn.Embedding(n_states, z_dim)
        self.bias = nn.Embedding(n_states, 1)
        nn.init.normal_(self.F.weight, std=0.1)
        nn.init.normal_(self.B.weight, std=0.1)
        nn.init.zeros_(self.bias.weight)

    def score(self, s: torch.Tensor, targets: torch.Tensor,
              jitter: float = 0.0) -> torch.Tensor:
        """f(s_i, t_j) for all i x j. s: (B,), targets: (K,) -> (B, K).
        `jitter` ignored (no coordinate space to perturb)."""
        return (self.F(s) @ self.B(targets).T) + self.bias(targets)[:, 0][None, :]

    def score_pairs(self, s: torch.Tensor, t: torch.Tensor,
                    jitter: float = 0.0) -> torch.Tensor:
        """f(s_i, t_i) elementwise. -> (B,)"""
        return (self.F(s) * self.B(t)).sum(-1) + self.bias(t)[:, 0]


class _MLP(nn.Module):
    def __init__(self, in_dim, z_dim, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, z_dim))

    def forward(self, x):
        return self.net(x)


class _CoordCritic(nn.Module):
    """MLP twin encoders on coordinates; last B output is the target bias.

    `jitter` adds fresh Gaussian input noise per call (train-time
    augmentation: a smoothness prior at the chosen scale; 0 at readout)."""

    def __init__(self, X: torch.Tensor, z_dim: int, hidden: int):
        super().__init__()
        self.register_buffer("X", X)
        self.Fnet = _MLP(X.shape[1], z_dim, hidden)
        self.Bnet = _MLP(X.shape[1], z_dim + 1, hidden)

    def _emb(self, idx, jitter=0.0):
        x = self.X[idx]
        if isinstance(jitter, torch.Tensor):
            x = x + jitter[idx][:, None] * torch.randn_like(x)
        elif jitter > 0:
            x = x + jitter * torch.randn_like(x)
        return x

    def score(self, s, targets, jitter: float = 0.0):
        Fz = self.Fnet(self._emb(s, jitter))
        Bz = self.Bnet(self._emb(targets, jitter))
        return Fz @ Bz[:, :-1].T + Bz[:, -1][None, :]

    def score_pairs(self, s, t, jitter: float = 0.0):
        Fz = self.Fnet(self._emb(s, jitter))
        Bz = self.Bnet(self._emb(t, jitter))
        return (Fz * Bz[:, :-1]).sum(-1) + Bz[:, -1]


@dataclass
class VMIResult:
    anchors: np.ndarray
    mi_bound: np.ndarray
    mi_plugin: np.ndarray
    H_plugin: np.ndarray
    n_pos: np.ndarray


class VMI:
    """InfoNCE/DV variational successor-MI field estimator (V1)."""

    def __init__(self, gamma: float = 0.97, z_dim: int = 64,
                 features: str = "tabular", hidden: int = 256,
                 n_epochs: int = 60, batch_size: int = 4096,
                 n_negatives: int = 256, lr: float = 1e-2,
                 lags_per_step: int = 4, holdout_frac: float = 0.25,
                 weight_decay: float = 0.0, aug_scale: float = 0.0,
                 device: str = "cuda:0", seed: int = 0):
        self.gamma = gamma
        self.z_dim = z_dim
        self.features = features
        self.hidden = hidden
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_negatives = n_negatives
        self.lr = lr
        self.lags_per_step = lags_per_step
        self.holdout_frac = holdout_frac
        self.weight_decay = weight_decay
        self.aug_scale = aug_scale
        self.device = device
        self.seed = seed

    def fit(self, traj: np.ndarray, n_states: int,
            X: np.ndarray | None = None) -> "VMI":
        """traj: (nt, T+1) int states in [0, n_states)."""
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        dev = self.device
        self.n_states = n_states
        a, p = harvest_pairs(traj, self.gamma, self.lags_per_step, rng)
        perm = rng.permutation(len(a))
        n_hold = int(self.holdout_frac * len(a))
        hold, tr = perm[:n_hold], perm[n_hold:]
        self._held_pairs = (a[hold], p[hold])
        A = torch.as_tensor(a[tr], dtype=torch.long, device=dev)
        P = torch.as_tensor(p[tr], dtype=torch.long, device=dev)
        if self.features == "tabular":
            self.net = _TabularCritic(n_states, self.z_dim).to(dev)
        else:
            Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32,
                                 device=dev)
            self.net = _CoordCritic(Xt, self.z_dim, self.hidden).to(dev)
        jit = 0.0
        if self.aug_scale > 0 and self.features == "coords":
            Xa = np.asarray(X)
            sub = np.arange(0, len(a), max(1, len(a) // 4096))
            jit = float(self.aug_scale * np.median(
                np.linalg.norm(Xa[p[sub]] - Xa[a[sub]], axis=1)))
        self.jitter_ = jit
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.n_epochs, eta_min=self.lr * 1e-2)
        n_pairs = len(A)
        g = torch.Generator(device="cpu").manual_seed(self.seed + 1)
        for _ in range(self.n_epochs):
            eperm = torch.randperm(n_pairs, generator=g).to(dev)
            for lo in range(0, n_pairs, self.batch_size):
                idx = eperm[lo:lo + self.batch_size]
                s, pos = A[idx], P[idx]
                neg = torch.randint(0, n_states, (self.n_negatives,),
                                    device=dev)
                f_pos = self.net.score_pairs(s, pos, jitter=jit)
                f_neg = self.net.score(s, neg, jitter=jit)   # (B, K)
                logits = torch.cat([f_pos[:, None], f_neg], dim=1)
                loss = nn.functional.cross_entropy(
                    logits, torch.zeros(len(s), dtype=torch.long,
                                        device=dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        return self

    @torch.no_grad()
    def mi_field(self, anchors: np.ndarray) -> VMIResult:
        dev = self.device
        n = self.n_states
        anchors = np.asarray(anchors)
        At = torch.as_tensor(anchors, dtype=torch.long, device=dev)
        allst = torch.arange(n, dtype=torch.long, device=dev)
        Fall = self.net.score(At, allst)                     # (A, n)
        logZ = torch.logsumexp(Fall, dim=1) - np.log(n)      # log E_unif e^f
        Pm = torch.softmax(Fall, dim=1)
        H_plug = -(Pm * torch.log(Pm.clamp_min(1e-30))).sum(1)
        mi_plug = np.log(n) - H_plug
        # DV bound: mean f over HELD-OUT future visits of each anchor - logZ
        a, p = self._held_pairs
        sums = np.zeros(n)
        cnts = np.zeros(n)
        Ait = torch.as_tensor(a, dtype=torch.long, device=dev)
        Pit = torch.as_tensor(p, dtype=torch.long, device=dev)
        f_ap = np.empty(len(a))
        for lo in range(0, len(a), 262144):
            hi = min(lo + 262144, len(a))
            f_ap[lo:hi] = self.net.score_pairs(
                Ait[lo:hi], Pit[lo:hi]).cpu().numpy()
        np.add.at(sums, a, f_ap)
        np.add.at(cnts, a, 1)
        mean_f = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
        mi_bound = mean_f[anchors] - logZ.cpu().numpy()
        n_pos = cnts[anchors]
        mi_bound = np.where(n_pos >= 3, mi_bound, np.nan)
        return VMIResult(anchors=anchors, mi_bound=mi_bound,
                         mi_plugin=mi_plug.cpu().numpy(),
                         H_plugin=H_plug.cpu().numpy(), n_pos=n_pos)


def vmi_ensemble(traj, n_states, anchors, seeds=(0, 1), X=None,
                 **kw) -> VMIResult:
    """Estimate-level seed ensemble (mean of fields, never of critics)."""
    outs = [VMI(seed=s, **kw).fit(traj, n_states, X=X).mi_field(anchors)
            for s in seeds]
    return VMIResult(
        anchors=np.asarray(anchors),
        mi_bound=np.nanmean([o.mi_bound for o in outs], axis=0),
        mi_plugin=np.mean([o.mi_plugin for o in outs], axis=0),
        H_plugin=np.mean([o.H_plugin for o in outs], axis=0),
        n_pos=outs[0].n_pos)


# ---------------------------------------------------------------------------
# V5 — Mohamed/Barber–Agakov action-decoder estimator
# ---------------------------------------------------------------------------

def harvest_paths(traj: np.ndarray, gamma: float, lags_per_step: int,
                  rng: np.random.Generator):
    """Sample variable-length paths (s_t, ..., s_{t+n}), n ~ Geom(1-gamma),
    truncated at walk ends. Returns flat step-arrays:
      cur, nxt, goal : (n_steps,) int  — per-step training triples
      path_id        : (n_steps,) int  — which path each step belongs to
      path_anchor    : (n_paths,) int  — start state of each path
    """
    nt, T1 = traj.shape
    T = T1 - 1
    cur, nxt, goal, rem, pid, anchors = [], [], [], [], [], []
    p_counter = 0
    for _ in range(lags_per_step):
        n = rng.geometric(p=1.0 - gamma, size=nt * T)
        t_idx = np.tile(np.arange(T), nt)
        w_idx = np.repeat(np.arange(nt), T)
        ok = t_idx + n <= T
        for w, t, ln in zip(w_idx[ok], t_idx[ok], n[ok]):
            seg = traj[w, t:t + ln + 1]
            cur.append(seg[:-1])
            nxt.append(seg[1:])
            goal.append(np.full(ln, seg[-1]))
            rem.append(np.arange(ln, 0, -1))
            pid.append(np.full(ln, p_counter))
            anchors.append(seg[0])
            p_counter += 1
    return (np.concatenate(cur), np.concatenate(nxt), np.concatenate(goal),
            np.concatenate(rem), np.concatenate(pid), np.asarray(anchors))


MAX_REM = 32


class _GoalDecoder(nn.Module):
    """Tabular goal-conditioned bridge decoder. The true bridge
    p(j|i,g,rem) ~ P(i,j) * P^{rem-1}(j,g) needs a genuine 3-way interaction
    (goal attraction modulated by remaining steps), so the goal-conditioned
    head is an MLP over [U[i], V[g], W[rem]]; the unconditioned pi-hat is the
    plain bilinear one-step model."""

    def __init__(self, n_states: int, z_dim: int, goal_conditioned: bool,
                 hidden: int = 128):
        super().__init__()
        self.U = nn.Embedding(n_states, z_dim)
        self.V = nn.Embedding(n_states, z_dim) if goal_conditioned else None
        self.W = (nn.Embedding(MAX_REM + 1, z_dim)
                  if goal_conditioned else None)
        self.head = (nn.Sequential(
            nn.Linear(3 * z_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, z_dim)) if goal_conditioned else None)
        self.E = nn.Embedding(n_states, z_dim)
        self.b = nn.Embedding(n_states, 1)
        for emb in (self.U, self.E) + (
                (self.V, self.W) if goal_conditioned else ()):
            nn.init.normal_(emb.weight, std=0.1)
        nn.init.zeros_(self.b.weight)

    def log_prob(self, cur, nxt, goal=None, rem=None):
        u = self.U(cur)
        if self.V is not None:
            h = torch.cat([u, self.V(goal),
                           self.W(rem.clamp(max=MAX_REM))], dim=-1)
            u = u + self.head(h)
        logits = u @ self.E.weight.T + self.b.weight[:, 0][None, :]
        return -nn.functional.cross_entropy(logits, nxt, reduction="none")


@dataclass
class BAResult:
    anchors: np.ndarray
    ba: np.ndarray          # BA bound field: E_paths[log q - log pi_hat]
    capacity: np.ndarray    # log Z(s): logmeanexp over paths of the same
    n_paths: np.ndarray


class BADecoder:
    """V5 — Mohamed/BA estimator: per-anchor decodability gain of paths."""

    def __init__(self, gamma: float = 0.9, z_dim: int = 64,
                 n_epochs: int = 30, batch_size: int = 8192,
                 lr: float = 1e-2, lags_per_step: int = 4,
                 holdout_frac: float = 0.5,
                 device: str = "cuda:0", seed: int = 0):
        self.gamma = gamma
        self.z_dim = z_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lags_per_step = lags_per_step
        self.holdout_frac = holdout_frac
        self.device = device
        self.seed = seed

    def fit(self, traj: np.ndarray, n_states: int) -> "BADecoder":
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        dev = self.device
        self.n_states = n_states
        cur, nxt, goal, rem, pid, anchors = harvest_paths(
            traj, self.gamma, self.lags_per_step, rng)
        n_paths = len(anchors)
        held_paths = rng.random(n_paths) < self.holdout_frac
        held_step = held_paths[pid]
        self._held = (cur[held_step], nxt[held_step], goal[held_step],
                      rem[held_step], pid[held_step], anchors)
        tr = ~held_step
        C = torch.as_tensor(cur[tr], dtype=torch.long, device=dev)
        Nx = torch.as_tensor(nxt[tr], dtype=torch.long, device=dev)
        G = torch.as_tensor(goal[tr], dtype=torch.long, device=dev)
        R = torch.as_tensor(rem[tr], dtype=torch.long, device=dev)
        self.q = _GoalDecoder(n_states, self.z_dim,
                              goal_conditioned=True).to(dev)
        self.pi = _GoalDecoder(n_states, self.z_dim,
                               goal_conditioned=False).to(dev)
        params = list(self.q.parameters()) + list(self.pi.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.n_epochs, eta_min=self.lr * 1e-2)
        n_tr = len(C)
        g = torch.Generator(device="cpu").manual_seed(self.seed + 1)
        for _ in range(self.n_epochs):
            perm = torch.randperm(n_tr, generator=g).to(dev)
            for lo in range(0, n_tr, self.batch_size):
                idx = perm[lo:lo + self.batch_size]
                lq = self.q.log_prob(C[idx], Nx[idx], G[idx], R[idx])
                lp = self.pi.log_prob(C[idx], Nx[idx])
                loss = -(lq.mean() + lp.mean())
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        return self

    @torch.no_grad()
    def ba_field(self, anchors: np.ndarray) -> BAResult:
        dev = self.device
        cur, nxt, goal, rem, pid, path_anchor = self._held
        lr_step = np.empty(len(cur))
        Ct = torch.as_tensor(cur, dtype=torch.long, device=dev)
        Nt = torch.as_tensor(nxt, dtype=torch.long, device=dev)
        Gt = torch.as_tensor(goal, dtype=torch.long, device=dev)
        Rt = torch.as_tensor(rem, dtype=torch.long, device=dev)
        for lo in range(0, len(cur), 262144):
            hi = min(lo + 262144, len(cur))
            lq = self.q.log_prob(Ct[lo:hi], Nt[lo:hi], Gt[lo:hi], Rt[lo:hi])
            lp = self.pi.log_prob(Ct[lo:hi], Nt[lo:hi])
            lr_step[lo:hi] = (lq - lp).cpu().numpy()
        # per-path sums
        n_paths = len(path_anchor)
        path_sum = np.zeros(n_paths)
        np.add.at(path_sum, pid, lr_step)
        held_ids = np.unique(pid)
        # per-anchor-state aggregation over held-out paths
        n = self.n_states
        ba_sum = np.zeros(n)
        ba_cnt = np.zeros(n)
        anchor_of = path_anchor[held_ids]
        vals = path_sum[held_ids]
        np.add.at(ba_sum, anchor_of, vals)
        np.add.at(ba_cnt, anchor_of, 1)
        ba = np.where(ba_cnt >= 3, ba_sum / np.maximum(ba_cnt, 1), np.nan)
        # capacity: log mean exp per anchor (clipped for stability)
        cap = np.full(n, np.nan)
        order = np.argsort(anchor_of)
        sa, sv = anchor_of[order], np.clip(vals[order], -30, 30)
        bounds = np.searchsorted(sa, np.arange(n))
        bounds = np.append(bounds, len(sa))
        for st in np.unique(sa):
            seg = sv[bounds[st]:bounds[st + 1]]
            if len(seg) >= 3:
                mx = seg.max()
                cap[st] = mx + np.log(np.mean(np.exp(seg - mx)))
        anchors = np.asarray(anchors)
        return BAResult(anchors=anchors, ba=ba[anchors],
                        capacity=cap[anchors], n_paths=ba_cnt[anchors])


def ba_ensemble(traj, n_states, anchors, seeds=(0, 1), **kw) -> BAResult:
    outs = [BADecoder(seed=s, **kw).fit(traj, n_states).ba_field(anchors)
            for s in seeds]
    return BAResult(
        anchors=np.asarray(anchors),
        ba=np.nanmean([o.ba for o in outs], axis=0),
        capacity=np.nanmean([o.capacity for o in outs], axis=0),
        n_paths=outs[0].n_paths)


# ---------------------------------------------------------------------------
# V2 — first-action empowerment field (action-conditioned, slip-noise-aware)
# ---------------------------------------------------------------------------

@dataclass
class FAMIResult:
    anchors: np.ndarray
    fami: np.ndarray        # BA bound on I(a_t; s_{t+k} | s_t)
    n_pos: np.ndarray


class FirstActionMI:
    """V2 — variational lower bound on the first-action empowerment field
    I(a_t; s_{t+k}|s_t), k ~ Geom(1-gamma), via inverse-dynamics BA:
      I >= E[log q(a|s, s+) - log pi_hat(a|s)].
    Actions are INTENDED next states (may differ from realized under slip
    noise); both models trained by MLE, no negatives. Under deterministic
    execution this field equals the one-step-ahead future MI; under
    heteroskedastic slip noise it counts only CONTROLLABLE influence."""

    def __init__(self, gamma: float = 0.9, z_dim: int = 64,
                 n_epochs: int = 30, batch_size: int = 8192,
                 lr: float = 1e-2, lags_per_step: int = 8,
                 holdout_frac: float = 0.5,
                 device: str = "cuda:0", seed: int = 0):
        self.gamma = gamma
        self.z_dim = z_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.lags_per_step = lags_per_step
        self.holdout_frac = holdout_frac
        self.device = device
        self.seed = seed

    def fit(self, traj: np.ndarray, intended: np.ndarray,
            n_states: int) -> "FirstActionMI":
        """traj: (nt, T+1) realized states; intended: (nt, T) intended
        next-state per step (equals traj[:,1:] when no slip)."""
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        dev = self.device
        self.n_states = n_states
        nt, T1 = traj.shape
        T = T1 - 1
        t_idx = np.tile(np.arange(T), nt)
        w_idx = np.repeat(np.arange(nt), T)
        s_l, a_l, g_l = [], [], []
        for _ in range(self.lags_per_step):
            k = rng.geometric(p=1.0 - self.gamma, size=t_idx.size)
            ok = t_idx + k <= T
            s_l.append(traj[w_idx[ok], t_idx[ok]])
            a_l.append(intended[w_idx[ok], t_idx[ok]])
            g_l.append(traj[w_idx[ok], t_idx[ok] + k[ok]])
        s, a, g = (np.concatenate(s_l), np.concatenate(a_l),
                   np.concatenate(g_l))
        perm = rng.permutation(len(s))
        n_hold = int(self.holdout_frac * len(s))
        hold, tr = perm[:n_hold], perm[n_hold:]
        self._held = (s[hold], a[hold], g[hold])
        St = torch.as_tensor(s[tr], dtype=torch.long, device=dev)
        At = torch.as_tensor(a[tr], dtype=torch.long, device=dev)
        Gt = torch.as_tensor(g[tr], dtype=torch.long, device=dev)
        R1 = torch.ones(len(St), dtype=torch.long, device=dev)
        self.q = _GoalDecoder(n_states, self.z_dim,
                              goal_conditioned=True).to(dev)
        self.pi = _GoalDecoder(n_states, self.z_dim,
                               goal_conditioned=False).to(dev)
        params = list(self.q.parameters()) + list(self.pi.parameters())
        opt = torch.optim.Adam(params, lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.n_epochs, eta_min=self.lr * 1e-2)
        n_tr = len(St)
        gcpu = torch.Generator(device="cpu").manual_seed(self.seed + 1)
        for _ in range(self.n_epochs):
            eperm = torch.randperm(n_tr, generator=gcpu).to(dev)
            for lo in range(0, n_tr, self.batch_size):
                idx = eperm[lo:lo + self.batch_size]
                lq = self.q.log_prob(St[idx], At[idx], Gt[idx],
                                     R1[idx])
                lp = self.pi.log_prob(St[idx], At[idx])
                loss = -(lq.mean() + lp.mean())
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        return self

    @torch.no_grad()
    def fami_field(self, anchors: np.ndarray) -> FAMIResult:
        dev = self.device
        n = self.n_states
        s, a, g = self._held
        St = torch.as_tensor(s, dtype=torch.long, device=dev)
        At = torch.as_tensor(a, dtype=torch.long, device=dev)
        Gt = torch.as_tensor(g, dtype=torch.long, device=dev)
        lr_v = np.empty(len(s))
        for lo in range(0, len(s), 262144):
            hi = min(lo + 262144, len(s))
            r1 = torch.ones(hi - lo, dtype=torch.long, device=dev)
            lq = self.q.log_prob(St[lo:hi], At[lo:hi], Gt[lo:hi], r1)
            lp = self.pi.log_prob(St[lo:hi], At[lo:hi])
            lr_v[lo:hi] = (lq - lp).cpu().numpy()
        sums = np.zeros(n)
        cnts = np.zeros(n)
        np.add.at(sums, s, lr_v)
        np.add.at(cnts, s, 1)
        fami = np.where(cnts >= 3, sums / np.maximum(cnts, 1), np.nan)
        anchors = np.asarray(anchors)
        return FAMIResult(anchors=anchors, fami=fami[anchors],
                          n_pos=cnts[anchors])


def fami_ensemble(traj, intended, n_states, anchors,
                  seeds=(0, 1), **kw) -> FAMIResult:
    outs = [FirstActionMI(seed=s, **kw)
            .fit(traj, intended, n_states).fami_field(anchors)
            for s in seeds]
    return FAMIResult(
        anchors=np.asarray(anchors),
        fami=np.nanmean([o.fami for o in outs], axis=0),
        n_pos=outs[0].n_pos)


# ---------------------------------------------------------------------------
# TD-InfoNCE — bootstrapped contrastive successor critic (continuous-ready)
# ---------------------------------------------------------------------------

class TDInfoNCE:
    """Temporal-difference InfoNCE (Zheng & Eysenbach style): the critic
    f(s, s') is trained toward the discounted occupancy density ratio
    log[M_gamma(s,·)/rho(·)] using ONE-STEP pairs only — the gamma-horizon is
    built by bootstrapping:
      L = (1-gamma) * CE(logits(s, [s'; cands]), 0)
        + gamma     * CE(logits(s, cands), softmax(f_tgt(s', cands)))
    with a Polyak-averaged target critic. This re-imports the Markov
    structure that flat lag-pair InfoNCE forfeits."""

    def __init__(self, gamma: float = 0.95, z_dim: int = 64,
                 features: str = "coords", hidden: int = 256,
                 n_epochs: int = 150, batch_size: int = 4096,
                 n_candidates: int = 511, lr: float = 1e-3,
                 tau_polyak: float = 0.01, holdout_frac: float = 0.5,
                 lags_per_step: int = 8, aug_scale: float = 0.0,
                 aug_nugget: bool | str = False, aug_anneal: bool = False,
                 holdout_phase: int | None = None, lam: float = 0.0,
                 min_lag: int = 1, cv_threshold: float = 0.3,
                 device: str = "cuda:0", seed: int = 0):
        self.gamma = gamma
        self.z_dim = z_dim
        self.features = features
        self.hidden = hidden
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.n_candidates = n_candidates
        self.lr = lr
        self.tau_polyak = tau_polyak
        self.holdout_frac = holdout_frac
        self.lags_per_step = lags_per_step
        self.aug_scale = aug_scale     # x median gamma-lag displacement
        self.aug_nugget = aug_nugget   # subtract observation-noise floor
        self.aug_anneal = aug_anneal   # cosine 2x -> 0.25x over training
        self.holdout_phase = holdout_phase  # 0/1: complementary crossfit
        self.lam = lam                 # TD(lambda): weight of real-lag MC
        self.min_lag = min_lag         # decorrelate from AR observation noise
        self.cv_threshold = cv_threshold  # aug_nugget="auto" switch point
        self.device = device
        self.seed = seed

    def fit(self, traj: np.ndarray, n_states: int,
            X: np.ndarray | None = None) -> "TDInfoNCE":
        import copy
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        dev = self.device
        self.n_states = n_states
        # aug_nugget="auto": detect structured noise (high spatial CV of the
        # local nugget) -> disable the subtraction (structured noise wants
        # over-smoothing); otherwise use the soft nugget.
        nugget_mode = self.aug_nugget
        self.cv_ = np.nan
        if nugget_mode == "auto" and self.features == "coords":
            self.cv_ = noise_structure_cv(np.asarray(X), traj,
                                          rng=self.seed + 5)
            nugget_mode = False if self.cv_ > self.cv_threshold else "soft"
        self._nugget_mode = nugget_mode
        # geometric-lag pairs: held half -> DV readout; train half -> TD(lam)
        a, p = harvest_pairs(traj, self.gamma, self.lags_per_step, rng,
                             min_lag=self.min_lag)
        if self.holdout_phase is None:
            keep = rng.random(len(a)) < self.holdout_frac
        else:
            # fixed split shared across net seeds -> complementary phases
            mask = np.random.default_rng(9999).random(len(a)) < 0.5
            keep = mask if self.holdout_phase == 0 else ~mask
        self._held_pairs = (a[keep], p[keep])
        a_tr, p_tr = a[~keep], p[~keep]
        # training data: consecutive pairs
        s = traj[:, :-1].ravel()
        sp = traj[:, 1:].ravel()
        St = torch.as_tensor(s, dtype=torch.long, device=dev)
        Spt = torch.as_tensor(sp, dtype=torch.long, device=dev)
        if self.features == "tabular":
            self.net = _TabularCritic(n_states, self.z_dim).to(dev)
        else:
            Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32,
                                 device=dev)
            self.net = _CoordCritic(Xt, self.z_dim, self.hidden).to(dev)
        tgt = copy.deepcopy(self.net)
        for prm in tgt.parameters():
            prm.requires_grad_(False)
        # augmentation scale: aug_scale x median gamma-lag displacement —
        # smooth away structure below the horizon scale. With aug_nugget,
        # subtract the observation-noise floor (variogram nugget: linear
        # extrapolation of MSD(k) to k=0 estimates 2*sigma_obs^2) so already-
        # noisy data receives little ADDED jitter.
        jit = 0.0
        if self.aug_scale > 0 and self.features == "coords":
            Xa = np.asarray(X)
            sub = np.arange(0, len(a), max(1, len(a) // 4096))
            lag2 = np.median(np.linalg.norm(Xa[p[sub]] - Xa[a[sub]],
                                            axis=1))**2
            if nugget_mode:
                nt_, T1 = traj.shape
                w = rng.integers(0, nt_, 4096)
                t = rng.integers(0, T1 - 2, 4096)
                d1 = ((Xa[traj[w, t + 1]] - Xa[traj[w, t]])**2).sum(1)
                d2 = ((Xa[traj[w, t + 2]] - Xa[traj[w, t]])**2).sum(1)
                sig2 = max(0.0, float(np.mean(d1) - (np.mean(d2)
                                                     - np.mean(d1)))) / 2
                if nugget_mode == "soft":
                    # scale the subtraction by the noise-dominance ratio
                    # rho = 2sig^2/lag^2: full nugget when noise dominates
                    # (hd64), ~base jitter when geometry dominates (hetero)
                    rho = min(1.0, 2 * sig2 / max(lag2, 1e-12))
                    lag2 = max(lag2 - 2 * sig2 * rho, 0.01 * lag2)
                else:
                    lag2 = max(lag2 - 2 * sig2, 0.01 * lag2)
            jit = float(self.aug_scale * np.sqrt(lag2))
            if self.aug_nugget == "equalize":
                # per-state compensating jitter: homogenize the total noise
                # across the manifold (structured sigma(x) -> iso), then add
                # the geometric smoothing on top
                from scipy.spatial import cKDTree as _KD
                nt_, T1 = traj.shape
                w = rng.integers(0, nt_, 20000)
                t = rng.integers(0, T1 - 2, 20000)
                x0 = Xa[traj[w, t]]
                d1 = ((Xa[traj[w, t + 1]] - x0)**2).sum(1)
                d2 = ((Xa[traj[w, t + 2]] - x0)**2).sum(1)
                ptree = _KD(x0)
                _, nn_ = ptree.query(Xa, k=64)
                loc = np.maximum(
                    0.0, d1[nn_].mean(1) - (d2[nn_].mean(1)
                                            - d1[nn_].mean(1))) / 2
                sig2_tgt = np.quantile(loc, 0.9)
                geo2 = max(lag2 - 2 * sig2_tgt, 0.01 * lag2)
                jvec = np.sqrt(np.maximum(sig2_tgt - loc, 0.0)
                               + (self.aug_scale**2) * geo2)
                jit = torch.as_tensor(jvec, dtype=torch.float32,
                                      device=dev)
        self.jitter_ = (float(jit.mean()) if isinstance(jit, torch.Tensor)
                        else jit)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.n_epochs, eta_min=self.lr * 1e-2)
        n_tr = len(St)
        use_mc = self.lam > 0 and len(a_tr) > 0
        if use_mc:
            Amc = torch.as_tensor(a_tr, dtype=torch.long, device=dev)
            Pmc = torch.as_tensor(p_tr, dtype=torch.long, device=dev)
        g = torch.Generator(device="cpu").manual_seed(self.seed + 1)
        for ep in range(self.n_epochs):
            if (self.aug_anneal and not isinstance(jit, torch.Tensor)
                    and jit > 0):
                frac = ep / max(self.n_epochs - 1, 1)
                jit_e = jit * (0.25 + (2.0 - 0.25)
                               * 0.5 * (1 + np.cos(np.pi * frac)))
            else:
                jit_e = jit
            perm = torch.randperm(n_tr, generator=g).to(dev)
            for lo in range(0, n_tr, self.batch_size):
                idx = perm[lo:lo + self.batch_size]
                si, spi = St[idx], Spt[idx]
                cand = torch.randint(0, n_states, (self.n_candidates,),
                                     device=dev)
                f_next = self.net.score_pairs(si, spi, jitter=jit_e)
                f_cand = self.net.score(si, cand, jitter=jit_e)     # (B, K)
                logits1 = torch.cat([f_next[:, None], f_cand], dim=1)
                loss1 = nn.functional.cross_entropy(
                    logits1, torch.zeros(len(si), dtype=torch.long,
                                         device=dev))
                with torch.no_grad():
                    p_tgt = torch.softmax(
                        tgt.score(spi, cand, jitter=jit_e), dim=1)
                loss2 = -(p_tgt
                          * torch.log_softmax(f_cand, dim=1)).sum(1).mean()
                loss = (1 - self.gamma) * loss1 + self.gamma * loss2
                if use_mc:
                    # TD(lambda) grounding: real Geom(gamma)-lag positives
                    midx = torch.randint(0, len(Amc), (len(si),),
                                         device=dev)
                    ami, pmi = Amc[midx], Pmc[midx]
                    f_mcp = self.net.score_pairs(ami, pmi, jitter=jit_e)
                    f_mcc = self.net.score(ami, cand, jitter=jit_e)
                    lmc = nn.functional.cross_entropy(
                        torch.cat([f_mcp[:, None], f_mcc], dim=1),
                        torch.zeros(len(si), dtype=torch.long, device=dev))
                    loss = (1 - self.lam) * loss + self.lam * lmc
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    for pn, pt in zip(self.net.parameters(),
                                      tgt.parameters()):
                        pt.mul_(1 - self.tau_polyak).add_(
                            pn, alpha=self.tau_polyak)
            sched.step()
        return self

    # readout shares VMI's semantics; reuse its implementation
    mi_field = VMI.mi_field


# ---------------------------------------------------------------------------
# specSENT — spectral successor features: learned eigenbasis + analytic
# resolvent (no bootstrap, no softmax temperature, horizon = post-hoc knob)
# ---------------------------------------------------------------------------

class SpectralSF:
    """Learn the top-k eigenfunctions of the transition operator via the
    spectral contrastive loss (HaoChen et al.) on ONE-STEP pairs:
        L = -2 E[phi(x)^T phi(x')] + E_{y ~ marginal}[(phi(x)^T phi(y))^2]
    then rotate into the eigenbasis by solving the generalized eigenproblem
    A w = lambda B w with A = E[phi(x) phi(x')^T]_sym, B = E[phi phi^T],
    and reconstruct the successor measure ANALYTICALLY:
        M_gamma(s, y) ~ sum_i (1-g)lam_i/(1-g lam_i) psi_i(s) psi_i(y)
    Readout: clipped+normalized rows over a corpus subsample -> entropy
    (orientation -H). Smoothness of eigenfunctions gives built-in robustness
    to structured corruption; input jitter reuses the td4 auto rule."""

    def __init__(self, gamma: float = 0.9, k_eig: int = 64,
                 features: str = "coords", hidden: int = 256,
                 n_epochs: int = 200, batch_size: int = 4096,
                 lr: float = 1e-3, aug_scale: float = 0.5,
                 aug_nugget: bool | str = "auto", cv_threshold: float = 0.3,
                 ridge: float = 1e-4,
                 device: str = "cuda:0", seed: int = 0):
        self.gamma = gamma
        self.k_eig = k_eig
        self.features = features
        self.hidden = hidden
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.aug_scale = aug_scale
        self.aug_nugget = aug_nugget
        self.cv_threshold = cv_threshold
        self.ridge = ridge
        self.device = device
        self.seed = seed

    def fit(self, traj: np.ndarray, n_states: int,
            X: np.ndarray | None = None) -> "SpectralSF":
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        dev = self.device
        self.n_states = n_states
        s = traj[:, :-1].ravel()
        sp = traj[:, 1:].ravel()
        # jitter via the td4 auto rule (lag scale at the gamma horizon)
        jit = 0.0
        if self.aug_scale > 0 and self.features == "coords":
            Xa = np.asarray(X)
            a_, p_ = harvest_pairs(traj, self.gamma, 2, rng)
            sub = np.arange(0, len(a_), max(1, len(a_) // 4096))
            lag2 = np.median(np.linalg.norm(Xa[p_[sub]] - Xa[a_[sub]],
                                            axis=1))**2
            mode = self.aug_nugget
            self.cv_ = np.nan
            if mode == "auto":
                self.cv_ = noise_structure_cv(Xa, traj, rng=self.seed + 5)
                mode = False if self.cv_ > self.cv_threshold else "soft"
            if mode:
                w = rng.integers(0, traj.shape[0], 4096)
                t = rng.integers(0, traj.shape[1] - 2, 4096)
                d1 = ((Xa[traj[w, t + 1]] - Xa[traj[w, t]])**2).sum(1)
                d2 = ((Xa[traj[w, t + 2]] - Xa[traj[w, t]])**2).sum(1)
                sig2 = max(0.0, float(np.mean(d1)
                                      - (np.mean(d2) - np.mean(d1)))) / 2
                rho = min(1.0, 2 * sig2 / max(lag2, 1e-12))
                lag2 = max(lag2 - 2 * sig2 * rho, 0.01 * lag2)
            jit = float(self.aug_scale * np.sqrt(lag2))
        self.jitter_ = jit
        if self.features == "tabular":
            self.enc = nn.Embedding(n_states, self.k_eig).to(dev)
            nn.init.normal_(self.enc.weight, std=0.1)

            def phi(idx, jitter=0.0):
                return self.enc(idx)
        else:
            Xt = torch.as_tensor(np.asarray(X), dtype=torch.float32,
                                 device=dev)
            self.enc = _MLP(Xt.shape[1], self.k_eig, self.hidden).to(dev)
            self._X = Xt

            def phi(idx, jitter=0.0):
                x = Xt[idx]
                if jitter > 0:
                    x = x + jitter * torch.randn_like(x)
                return self.enc(x)
        self._phi = phi
        St = torch.as_tensor(s, dtype=torch.long, device=dev)
        Spt = torch.as_tensor(sp, dtype=torch.long, device=dev)
        opt = torch.optim.Adam(self.enc.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.n_epochs, eta_min=self.lr * 1e-2)
        n_tr = len(St)
        g = torch.Generator(device="cpu").manual_seed(self.seed + 1)
        for _ in range(self.n_epochs):
            perm = torch.randperm(n_tr, generator=g).to(dev)
            for lo in range(0, n_tr, self.batch_size):
                idx = perm[lo:lo + self.batch_size]
                f1 = phi(St[idx], jit)
                f2 = phi(Spt[idx], jit)
                # negatives from the OCCUPANCY marginal (required for
                # the spectral optimum; uniform negatives distort the basis
                # by a pi-ratio wherever occupancy is nonuniform)
                y = St[torch.randint(0, n_tr, (512,), device=dev)]
                fy = phi(y, jit)
                pos = -2.0 * (f1 * f2).sum(1).mean()
                neg = ((f1 @ fy.T)**2).mean()
                loss = pos + neg
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        # eigen-rotation: A w = lam B w on a large pair sample
        from scipy.linalg import eigh
        with torch.no_grad():
            m = min(100000, n_tr)
            ii = torch.as_tensor(rng.choice(n_tr, m, replace=False),
                                 device=dev)
            F1 = phi(St[ii]).cpu().numpy()
            F2 = phi(Spt[ii]).cpu().numpy()
        A = (F1.T @ F2 + F2.T @ F1) / (2 * m)
        Bm = (F1.T @ F1 + F2.T @ F2) / (2 * m)
        Bm += self.ridge * np.eye(self.k_eig)
        lam, W = eigh(A, Bm)
        self.lam_ = np.clip(lam, -0.999, 0.999)
        self.W_ = W
        return self

    @torch.no_grad()
    def sent_field(self, anchors: np.ndarray, corpus: np.ndarray,
                   gamma: float | None = None) -> np.ndarray:
        """-entropy of reconstructed resolvent rows over the corpus."""
        gamma = gamma or self.gamma
        dev = self.device
        At = torch.as_tensor(np.asarray(anchors), dtype=torch.long,
                             device=dev)
        Ct = torch.as_tensor(np.asarray(corpus), dtype=torch.long,
                             device=dev)
        Pa = self._phi(At).cpu().numpy() @ self.W_
        Pc = self._phi(Ct).cpu().numpy() @ self.W_
        gcoef = (1 - gamma) * self.lam_ / (1 - gamma * self.lam_)
        M = (Pa * gcoef[None, :]) @ Pc.T
        M = np.maximum(M, 0.0) + 1e-15
        M = M / M.sum(1, keepdims=True)
        return (M * np.log(M)).sum(1)          # = -H, curvature orientation


def spectral_coords(est: "SpectralSF", n_states: int,
                    weight: str = "lam") -> np.ndarray:
    """Diffusion-map-style coordinates from a trained SpectralSF:
    Psi_hat(x) weighted per mode by lambda_hat (smooth modes emphasized).
    The robust use of learned spectra: as a denoised REPRESENTATION for a
    data-anchored estimator, not as an analytic resolvent (whose g(lambda)
    weights are hypersensitive near lambda=1; see exp-13 round 6)."""
    with torch.no_grad():
        idx = torch.arange(n_states, device=est.device)
        Psi = est._phi(idx).cpu().numpy() @ est.W_
    if weight == "lam":
        Psi = Psi * np.abs(est.lam_)[None, :]
    return Psi.astype(np.float32)
