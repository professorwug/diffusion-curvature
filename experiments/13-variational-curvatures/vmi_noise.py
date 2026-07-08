"""S3 — heteroskedastic slip noise: does action-conditioning rescue the
curvature field where entropy channels are fooled?

Environment: necklace d=3 kNN chain; intended step ~ P(s,.); with
probability eta(x) = eta_max*(1+u1)/2 (u1 = sphere-factor coordinate,
curvature-INDEPENDENT), the realized step is uniform over the 25-NN of s.
The noisy chain P_noisy = (1-eta)P + eta*U25.

Channels (paired, all at gamma=0.9, 5-anchor pooling):
  sent_exact  — exact resolvent entropy of P_noisy (entropy-family ceiling)
  I1_exact    — exact first-action MI field I(a; s+ | s) on P_noisy
  traceH0     — counts estimator of sent (realized walks only)
  v2 (fami)   — FirstActionMI estimator ((intended, realized) walks)

Pre-registered: as eta_max grows, r(sent_exact, ks) and r(traceH0, ks)
degrade (noise entropy contaminates); r(I1_exact, ks) degrades less; the v2
estimator tracks I1_exact.

Output: processed_data/vmi_noise.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent / "../11-successor-curvatures"))
from benchmark_kmetric_colosseum import affinity_from_D
from diffusion_curvature.menagerie import WarpedProduct, necklace_profile
from diffusion_curvature.variational import fami_ensemble

warnings.filterwarnings("ignore")

N = 1400
KNN = 10
KSLIP = 25
GAMMA = 0.9
DEV = "cuda:1" if torch.cuda.is_available() else "cpu"
NT, T = 40, 1600
NET_SEEDS = (0, 1, 2, 3)
POOL = 5
ETA_MAX = (0.0, 0.4, 0.8)


def sample_with_u(wp, n, rng):
    rng = np.random.default_rng(rng)
    pdf = np.maximum(wp.fg, 0) ** (wp.d - 1)
    cdf = np.cumsum(pdf)
    cdf = cdf / cdf[-1]
    r = np.interp(rng.random(n), cdf, wp.rg)
    u = rng.normal(size=(n, wp.d))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    cosang = np.clip(u @ u.T, -1, 1)
    ang = np.arccos(cosang)
    r1 = np.repeat(r, n).reshape(n, n)
    D = wp.pair_distance(r1.ravel(), r1.T.ravel(), ang.ravel()).reshape(n, n)
    D = 0.5 * (D + D.T)
    np.fill_diagonal(D, 0.0)
    return D, wp.scalar(r), u[:, 0]


def entropy_rows(R):
    R = np.clip(R, 1e-30, 1)
    return -(R * np.log(R)).sum(1)


def resolvent(P, device=DEV):
    n = P.shape[0]
    Pt = torch.as_tensor(P, dtype=torch.float64, device=device)
    M = (1 - GAMMA) * torch.linalg.inv(
        torch.eye(n, dtype=torch.float64, device=device) - GAMMA * Pt)
    M = np.maximum(M.cpu().numpy(), 0.0)
    return M / np.maximum(M.sum(1, keepdims=True), 1e-30)


def slip_walks(P, nbr25, eta, nt, T, seed):
    """Returns realized traj (nt,T+1) and intended next-states (nt,T)."""
    n = P.shape[0]
    rng = np.random.default_rng(seed)
    cum = np.cumsum(P, axis=1)
    traj = np.empty((nt, T + 1), dtype=int)
    intd = np.empty((nt, T), dtype=int)
    traj[:, 0] = rng.integers(0, n, nt)
    for t in range(T):
        cur = traj[:, t]
        u = rng.random(nt)
        a = np.array([np.searchsorted(cum[cur[w]], u[w])
                      for w in range(nt)])
        intd[:, t] = a
        slip = rng.random(nt) < eta[cur]
        rnd = nbr25[cur, rng.integers(0, KSLIP, nt)]
        traj[:, t + 1] = np.where(slip, rnd, a)
    return traj, intd


def exact_I1(P, Mres, eta, nbr25, anchors):
    """I1(s) = sum_a P(s,a) KL(B_a M || sum_a' P(s,a') B_a' M)."""
    out = np.empty(len(anchors))
    for q, s in enumerate(anchors):
        nz = np.nonzero(P[s])[0]
        U_row = Mres[nbr25[s]].mean(0)
        rows = (1 - eta[s]) * Mres[nz] + eta[s] * U_row[None, :]
        w = P[s, nz]
        marg = w @ rows
        rows_c = np.clip(rows, 1e-30, 1)
        marg_c = np.clip(marg, 1e-30, 1)
        kls = (rows_c * (np.log(rows_c) - np.log(marg_c)[None, :])).sum(1)
        out[q] = w @ kls
    return out


def main():
    f, L = necklace_profile(b=0.7)
    wp = WarpedProduct(f, L, d=3, periodic=True)
    rows = []
    for wseed in (0, 1):
        D, ks, u1 = sample_with_u(wp, N, wseed)
        s = np.median(D[np.triu_indices_from(D, 1)])
        D, ks = D / s, ks * s**2
        W = affinity_from_D(D, k=KNN)
        P = W / np.maximum(W.sum(1, keepdims=True), 1e-30)
        nbr25 = np.argsort(D, axis=1)[:, 1:KSLIP + 1]
        order = np.argsort(ks)
        eval_pts = order[(np.linspace(0.02, 0.98, 48) * (N - 1)).astype(int)]
        kt = ks[eval_pts]

        def r(v, ref):
            mm = np.isfinite(v) & np.isfinite(ref)
            return pearsonr(v[mm], ref[mm])[0] if mm.sum() > 8 else np.nan

        for eta_max in ETA_MAX:
            eta = eta_max * (1 + u1) / 2
            U25 = np.zeros((N, N))
            np.put_along_axis(U25, nbr25, 1.0 / KSLIP, axis=1)
            P_noisy = (1 - eta)[:, None] * P + eta[:, None] * U25
            Mres = resolvent(P_noisy)
            traj, intd = slip_walks(P, nbr25, eta, NT, T, wseed)
            V_idx, tl = np.unique(traj, return_inverse=True)
            tl = tl.reshape(traj.shape)
            nV = len(V_idx)
            # intended states relabeled: may include unvisited states
            all_states = np.unique(np.concatenate([V_idx, intd.ravel()]))
            lut = np.full(N, -1)
            lut[all_states] = np.arange(len(all_states))
            tl = lut[traj]
            il = lut[intd]
            nS = len(all_states)
            pool_idx_g = all_states[
                np.argsort(D[np.ix_(eval_pts, all_states)], axis=1)[:, :POOL]]
            anch_states_g = np.unique(pool_idx_g)
            loc = {st: q for q, st in enumerate(anch_states_g)}
            pool_q = np.vectorize(loc.get)(pool_idx_g)
            # exact channels
            H_ex = entropy_rows(Mres[anch_states_g])
            sent_f = np.nanmean((-H_ex)[pool_q], axis=1)
            I1_ex = exact_I1(P_noisy, Mres, eta, nbr25, anch_states_g)
            i1_f = np.nanmean(I1_ex[pool_q], axis=1)
            # counts estimator
            a, b = tl[:, :-1].ravel(), tl[:, 1:].ravel()
            C = sp.coo_matrix((np.ones(len(a)), (a, b)),
                              shape=(nS, nS)).tocsr()
            C = np.asarray((C + C.T).todense(), dtype=np.float64)
            Pe = (C + 1e-9) / (C + 1e-9).sum(1, keepdims=True)
            anch_pos = lut[anch_states_g]      # positions in all_states
            H_tr = entropy_rows(resolvent(Pe)[anch_pos])
            tr_f = np.nanmean((-H_tr)[pool_q], axis=1)
            # V2 estimator
            out = fami_ensemble(tl, il, nS, anch_pos, seeds=NET_SEEDS,
                                gamma=GAMMA, device=DEV)
            v2_f = np.nanmean(out.fami[pool_q], axis=1)
            # curvature orientation: entropy/spread channels are negated
            # (I1 and v2 are spread quantities like H; see V5 note)
            row = dict(wseed=wseed, eta_max=eta_max, nV=nV,
                       r_sent_ks=r(sent_f, kt),
                       r_I1_ks=r(-i1_f, kt),
                       r_traceH0_ks=r(tr_f, kt),
                       r_v2_ks=r(-v2_f, kt),
                       r_v2_I1=r(v2_f, i1_f),
                       r_I1_sent=r(i1_f, sent_f))
            rows.append(row)
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float)
                           else f"{k}={v}" for k, v in row.items()),
                  flush=True)
    df = pd.DataFrame(rows)
    Path("processed_data").mkdir(exist_ok=True)
    df.to_csv("processed_data/vmi_noise.csv", index=False)
    print("\n=== S3 noise summary (mean over walk seeds) ===")
    print(df.groupby("eta_max")[["r_sent_ks", "r_traceH0_ks", "r_I1_ks",
                                 "r_v2_ks", "r_v2_I1"]].mean().round(3))


if __name__ == "__main__":
    main()
