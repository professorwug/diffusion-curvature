"""Decompose specSENT's gate failure: k-truncation vs learning error."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parents[2]))
sys.path.insert(0, str(Path(__file__).parent / "tests"))
from test_13_variational_curvatures import (_exact_mi, _two_scale_chain,
                                            _walks)
from diffusion_curvature.variational import SpectralSF

GAMMA = 0.9
P, rng = _two_scale_chain()
n = P.shape[0]
traj = _walks(P, nt=20, T=800, rng=rng)
H_exact = np.log(n) - _exact_mi(P, GAMMA)

# true eigenpairs (reversible symmetrization)
lazy = np.diag(P).copy()
pi = 1.0 / (1.0 - lazy); pi /= pi.sum()
S = np.diag(pi**0.5) @ P @ np.diag(pi**-0.5)
S = 0.5 * (S + S.T)
lam, U = np.linalg.eigh(S)
order = np.argsort(-lam)
lam, U = lam[order], U[:, order]
psi = np.diag(pi**-0.5) @ U           # right eigenvectors, pi-orthonormal

corpus = np.random.default_rng(3).choice(traj.ravel(), 4000)
for k in (8, 16, 32, 79):
    g = (1 - GAMMA) * lam[:k] / (1 - GAMMA * lam[:k])
    M = (psi[:, :k] * g[None, :]) @ psi[corpus, :k].T
    M = np.maximum(M, 0) + 1e-15
    M /= M.sum(1, keepdims=True)
    Ht = -(M * np.log(M)).sum(1)
    print(f"exact-eig k={k}: r(H_trunc, H_exact) = "
          f"{np.corrcoef(Ht, H_exact)[0,1]:+.3f}")

est = SpectralSF(gamma=GAMMA, k_eig=16, features="tabular", n_epochs=150,
                 batch_size=2048, lr=1e-2, aug_scale=0.0, device="cpu",
                 seed=0).fit(traj, n)
print("learned lam top-8:", np.round(est.lam_[::-1][:8], 3))
print("true    lam top-8:", np.round(lam[:8], 3))
field = est.sent_field(np.arange(n), corpus)
print(f"learned k=16: r = {np.corrcoef(-field, H_exact)[0,1]:+.3f}")

# collision (Renyi-2) readout: C(s) = sum_i g_i^2 psi_i(s)^2, exact target
# = log sum_y M(s,y)^2 / pi(y)
Mex = (1 - GAMMA) * np.linalg.inv(np.eye(n) - GAMMA * P) @ P
C_true = np.log((Mex**2 / pi[None, :]).sum(1))
for k in (8, 16, 32, 79):
    g = (1 - GAMMA) * lam[:k] / (1 - GAMMA * lam[:k])
    C_k = np.log(((psi[:, :k]**2) * (g**2)[None, :]).sum(1))
    print(f"exact-eig collision k={k}: r vs C_true = "
          f"{np.corrcoef(C_k, C_true)[0,1]:+.3f}  |  r vs -H_exact = "
          f"{np.corrcoef(C_k, -H_exact)[0,1]:+.3f}")
# learned collision
g_l = (1 - GAMMA) * est.lam_ / (1 - GAMMA * est.lam_)
import torch
with torch.no_grad():
    Psi = est._phi(torch.arange(n)).numpy() @ est.W_
C_learn = np.log(np.maximum((Psi**2 * (g_l**2)[None, :]).sum(1), 1e-30))
print(f"learned collision k=16: r vs C_true = "
      f"{np.corrcoef(C_learn, C_true)[0,1]:+.3f}  |  vs -H_exact = "
      f"{np.corrcoef(C_learn, -H_exact)[0,1]:+.3f}")

# alignment + training-length sweep
def align(est):
    with torch.no_grad():
        Phi = est._phi(torch.arange(n)).numpy()
    Psi_l = Phi @ est.W_
    # pi-weighted projection of true top-8 onto learned span
    G = Psi_l * pi[:, None]**0.5
    Q, _ = np.linalg.qr(G)
    tru = psi[:, :8] * pi[:, None]**0.5
    proj = np.linalg.norm(Q.T @ tru, axis=0) / np.linalg.norm(tru, axis=0)
    return proj

for ep, kk, lr in ((150, 16, 1e-2), (600, 16, 1e-2), (600, 32, 3e-3)):
    e2 = SpectralSF(gamma=GAMMA, k_eig=kk, features="tabular", n_epochs=ep,
                    batch_size=2048, lr=lr, aug_scale=0.0, device="cpu",
                    seed=0).fit(traj, n)
    g2 = (1 - GAMMA) * e2.lam_ / (1 - GAMMA * e2.lam_)
    with torch.no_grad():
        Ps = e2._phi(torch.arange(n)).numpy() @ e2.W_
    C2 = np.log(np.maximum((Ps**2 * (g2**2)[None, :]).sum(1), 1e-30))
    pr = align(e2)
    print(f"ep={ep} k={kk} lr={lr}: r vs C_true="
          f"{np.corrcoef(C2, C_true)[0,1]:+.3f}  top8 proj="
          f"{np.round(pr, 2)}")

# multi-lag eigenvalue re-estimation (implied timescales)
def multilag_lambda(est, traj, lags=(1, 2, 4, 8, 16, 32)):
    with torch.no_grad():
        Phi = est._phi(torch.arange(n)).numpy()
    Psi_l = Phi @ est.W_
    k_eig = Psi_l.shape[1]
    lam_hat = np.full(k_eig, 0.0)
    T1 = traj.shape[1]
    for i in range(k_eig):
        f = Psi_l[traj]                      # (nt, T1, k)
        v = f[:, :, i]
        var = v.var()
        best = None
        for k in lags:
            a = np.mean(v[:, :-k] * v[:, k:]) / max(var, 1e-12)
            if a > 0.1:
                best = (a, k)
        if best is not None:
            lam_hat[i] = np.sign(best[0]) * abs(best[0]) ** (1.0 / best[1])
    return np.clip(lam_hat, -0.999, 0.999)

lam_ml = multilag_lambda(e2, traj)
g3 = (1 - GAMMA) * lam_ml / (1 - GAMMA * lam_ml)
with torch.no_grad():
    Ps3 = e2._phi(torch.arange(n)).numpy() @ e2.W_
C3 = np.log(np.maximum((Ps3**2 * (g3**2)[None, :]).sum(1), 1e-30))
print(f"multilag-lambda collision: r vs C_true = "
      f"{np.corrcoef(C3, C_true)[0,1]:+.3f}")
print("lam_ml top8:", np.round(np.sort(lam_ml)[::-1][:8], 4))
print("true   top8:", np.round(lam[:8], 4))

# long-lag rotation: diagonalize A_k at lag k=16 instead of one-step
from scipy.linalg import eigh as geigh
def longlag_rotation(est, traj, klag=16):
    with torch.no_grad():
        Phi = est._phi(torch.arange(n)).numpy()
    F = Phi[traj]                            # (nt, T1, k)
    F1 = F[:, :-klag].reshape(-1, F.shape[2])
    F2 = F[:, klag:].reshape(-1, F.shape[2])
    A = (F1.T @ F2 + F2.T @ F1) / (2 * len(F1))
    Fa = F.reshape(-1, F.shape[2])
    B = Fa.T @ Fa / len(Fa) + 1e-4 * np.eye(F.shape[2])
    lam_k, W = geigh(A, B)
    return W

W16 = longlag_rotation(e2, traj, 16)
with torch.no_grad():
    Psi4 = e2._phi(torch.arange(n)).numpy() @ W16
# multilag lambda on the new modes
T1 = traj.shape[1]
lam4 = np.zeros(Psi4.shape[1])
for i in range(Psi4.shape[1]):
    v = Psi4[traj][:, :, i]
    var = v.var()
    best = None
    for k in (1, 2, 4, 8, 16, 32):
        a = np.mean(v[:, :-k] * v[:, k:]) / max(var, 1e-12)
        if a > 0.1:
            best = (a, k)
    if best is not None:
        lam4[i] = np.sign(best[0]) * abs(best[0]) ** (1.0 / best[1])
lam4 = np.clip(lam4, -0.999, 0.999)
g4 = (1 - GAMMA) * lam4 / (1 - GAMMA * lam4)
C4 = np.log(np.maximum((Psi4**2 * (g4**2)[None, :]).sum(1), 1e-30))
print(f"longlag rotation + multilag lambda: r vs C_true = "
      f"{np.corrcoef(C4, C_true)[0,1]:+.3f}")

# CONTROL: true top-16 eigenvectors through the same rotation+multilag code
class FakeEst:
    def __init__(self, Psi_true):
        self._P = torch.as_tensor(Psi_true, dtype=torch.float32)
        self.W_ = np.eye(Psi_true.shape[1])
    def _phi(self, idx, jitter=0.0):
        return self._P[idx]

fk = FakeEst(psi[:, :16])
W16f = longlag_rotation(fk, traj, 16)
Psi5 = psi[:, :16] @ W16f
lam5 = np.zeros(16)
for i in range(16):
    v = Psi5[traj][:, :, i]
    var = v.var()
    best = None
    for k in (1, 2, 4, 8, 16, 32):
        a = np.mean(v[:, :-k] * v[:, k:]) / max(var, 1e-12)
        if a > 0.1:
            best = (a, k)
    if best is not None:
        lam5[i] = np.sign(best[0]) * abs(best[0]) ** (1.0 / best[1])
lam5 = np.clip(lam5, -0.999, 0.999)
g5 = (1 - GAMMA) * lam5 / (1 - GAMMA * lam5)
C5 = np.log(np.maximum((Psi5**2 * (g5**2)[None, :]).sum(1), 1e-30))
print(f"CONTROL true-psi through pipeline: r vs C_true = "
      f"{np.corrcoef(C5, C_true)[0,1]:+.3f}")
print("lam5:", np.round(np.sort(lam5)[::-1][:8], 4))
