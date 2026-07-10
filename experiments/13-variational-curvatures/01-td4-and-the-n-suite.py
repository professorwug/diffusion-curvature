import marimo

__generated_with = "0.23.13"
app = marimo.App()

with app.setup(hide_code=True):
    import marimo as mo


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # td4 and the N-suite
    ## Curvature fields from noisy continuous trajectories

    **The setting.** An agent diffuses over an unknown manifold; we observe only a
    *corrupted embedding* of its trajectory. Can we recover the manifold's scalar
    curvature field? This notebook illustrates the estimator that emerged from
    exp-13's iteration campaign — **td4**: a TD-InfoNCE critic with
    self-calibrating input-noise augmentation — and the benchmark that certifies
    it, the **N-suite**: Brownian motion on warped-product manifolds
    (analytic curvature everywhere) observed through six realistic noise models.

    The arc in one line: *variational lower bounds lost to counting on clean
    tabular data (S2), found their niche under observation noise (C-noise), and
    were then engineered into the best available instrument for noisy and
    high-dimensional regimes (rounds 1–6).*
    """)
    return


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch

    EXP = Path(".")                      # notebook lives in the experiment dir
    sys.path.insert(0, str(Path("../11-successor-curvatures")))

    from diffusion_curvature.menagerie import WarpedProduct, necklace_profile
    from diffusion_curvature.continuous_walks import brownian_walks, chi_embed
    from diffusion_curvature.variational import TDInfoNCE, noise_structure_cv

    DEV = "cuda:1" if torch.cuda.is_available() else "cpu"
    PD13 = EXP / "processed_data"

    return (
        DEV,
        PD13,
        TDInfoNCE,
        WarpedProduct,
        brownian_walks,
        chi_embed,
        necklace_profile,
        np,
        pd,
        plt,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## THE STAGE: warped products with analytic curvature

    The benchmark manifold is the **necklace** $S^1 \times_f S^{d-1}$ with profile
    $f(r) = 1 + b\cos(2\pi k r/L)$ — pearls joined by negatively-curved necks.
    Warped products give the scalar curvature *in closed form at every point*:

    $$R(r) = 2(d{-}1)\frac{-f''}{f} + (d{-}1)(d{-}2)\frac{1-f'^2}{f^2}$$

    Trajectories are exact Brownian motion, simulated in intrinsic coordinates
    via the radial SDE $dr = \tfrac{d-1}{2}\tfrac{f'}{f}dt + dW$ plus spherical
    BM with clock $dt/f(r)^2$. Every visited point carries its true $R$ — field
    ground truth with no anchor proxying.
    """)
    return


@app.cell(hide_code=True)
def _(WarpedProduct, necklace_profile, np):
    def build_manifold(b=0.7, k=2, d=3):
        f, L = necklace_profile(b=b, k=k)
        wp = WarpedProduct(f, L, d=d, periodic=True)
        m = wp.sample(1200, rng=42)
        scale = float(np.median(m["D"][np.triu_indices(1200, 1)]))
        return wp, scale


    wp3, s3 = build_manifold(d=3)
    return s3, wp3


@app.cell(hide_code=True)
def _(plt, s3, wp3):
    _fig, _ax = plt.subplots(1, 2, figsize=(9.5, 3), constrained_layout=True)
    _ax[0].plot(wp3.rg, wp3.fg, color="#2A6F97")
    _ax[0].set(xlabel="profile coordinate r", ylabel="f(r)",
               title="necklace profile (b=0.7, k=2)")
    _ax[1].plot(wp3.rg, wp3.R_g * s3**2, color="#C44536")
    _ax[1].axhline(0, color="gray", lw=0.6)
    _ax[1].set(xlabel="r", ylabel="scalar curvature R·s²",
               title="analytic curvature field (d=3, unit-scaled)")
    _fig
    return


@app.cell(hide_code=True)
def _(brownian_walks, chi_embed, s3, wp3):
    walks3 = brownian_walks(wp3, nt=12, T=1500, dt=2e-3 * s3**2, rng=0)
    chi3 = chi_embed(wp3, walks3["r"], walks3["u"]) / s3
    ks3 = walks3["ks"] * s3**2
    return chi3, ks3, walks3


@app.cell(hide_code=True)
def _(chi3, ks3, plt, walks3):
    _f2 = plt.figure(figsize=(9.5, 3.4), constrained_layout=True)
    _a1 = _f2.add_subplot(1, 2, 1, projection="3d")
    _pts = chi3.reshape(-1, chi3.shape[-1])
    _sc = _a1.scatter(_pts[:, 0], _pts[:, 1], _pts[:, 2], c=ks3.ravel(),
                      cmap="coolwarm", s=1.2, alpha=0.6)
    _a1.set_title("BM trajectories in the χ-embedding,\ncolored by true R")
    _a1.set_axis_off()
    _f2.colorbar(_sc, ax=_a1, shrink=0.7)
    _a2 = _f2.add_subplot(1, 2, 2)
    _a2.plot(walks3["r"][0][:400], lw=0.8, color="#2A6F97")
    _a2.set(xlabel="step", ylabel="r(t)", title="one walk's radial coordinate")
    _f2
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## SIX WAYS TO RUIN A MEASUREMENT: the noise models

    The estimator never sees χ — it sees $x_t = \mathrm{obs}(\chi(s_t))$ under one
    of six corruption models: **clean**; **iso05/iso15** (independent Gaussian at
    half / 1.7× the step scale); **hd64** (a random smooth lift into
    $\mathbb{R}^{64}$ plus noise — the learned-embedding analog); **hetero**
    (spatially varying $\sigma(x)$, curvature-independent by construction);
    **ar1** (temporally correlated noise, $\rho = 0.8$).
    """)
    return


@app.cell(hide_code=True)
def _(chi3, ks3, np, plt, walks3):
    from noise_benchmark import apply_noise

    def noisy_view(noise, seed=0):
        unit = dict(profile="nk2", d=3, wseed=seed, noise=noise)
        rngN = np.random.default_rng(7000 + seed)
        X, _ = apply_noise(unit, chi3, chi3[0, :1], walks3["u"][..., 0], rngN)
        return X


    _f3, _axs = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)
    for _ax3, _nz in zip(_axs, ("clean", "iso15", "hetero", "hd64")):
        _Xn = noisy_view(_nz)
        _ax3.scatter(_Xn[:, 0], _Xn[:, 2], c=ks3.ravel(), cmap="coolwarm",
                     s=1.0, alpha=0.5)
        _ax3.set_title(_nz)
        _ax3.set_axis_off()
    _f3.suptitle("what the estimator actually sees (two observed coords, true-R colors)")
    _f3
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## THE INSTRUMENT: td4

    **Critic.** Twin MLPs score pairs, $f(x, y) = F(x)^\top B(y) + b(y)$, trained
    by **TD-InfoNCE** (Zheng & Eysenbach): one-step pairs only, with the
    $\gamma$-horizon built by bootstrap,

    $$\mathcal{L} = (1{-}\gamma)\,\mathrm{CE}\big(f(s,[s'; \text{cands}]),\, s'\big)
     + \gamma\,\mathrm{CE}\big(f(s,\text{cands}),\, \mathrm{softmax}\, f_{\text{tgt}}(s', \text{cands})\big).$$

    At optimum $e^{f} \propto M_\gamma(s,\cdot)/\rho(\cdot)$ — the discounted
    occupancy density ratio. **Readout** (Donsker–Varadhan, data-anchored): held-out
    $\gamma$-lag pairs give $\hat I(s) = \mathbb{E}[f(s, s^+)] - \log\mathbb{E}_\rho[e^{f(s,\cdot)}]$,
    pooled over the 32 nearest observed states. Orientation: concentration = positive curvature.

    **The self-calibration stack** (all from data, no ground truth anywhere):

    1. **Jitter**: fresh input noise each batch at $0.5\times$ the median $\gamma$-lag
       displacement — the smoothness prior that fixes clean-data overfitting.
    2. **Soft nugget**: subtract the observation-noise floor (variogram intercept),
       scaled by the noise-dominance ratio — protects hd64 without touching hetero.
    3. **CV switch**: the spatial coefficient of variation of the *local* nugget
       detects structured noise (hetero 0.64, iso ≤ 0.12; threshold 0.3) and
       disables the subtraction where over-smoothing is wanted.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.mermaid("""
    graph LR
        T[trajectories x_t] -->|one-step pairs| L[TD-InfoNCE loss]
        T -->|gamma-lag pairs| J[jitter rule<br/>0.5 x lag scale]
        T --> CV[CV noise-structure<br/>detector]
        CV -->|structured| J2[base jitter]
        CV -->|unstructured| J3[soft nugget]
        J2 --> L
        J3 --> L
        L --> C[critic f = F.B + b]
        T -->|held-out lag pairs| DV[DV readout, pooled]
        C --> DV
        DV --> K[curvature field]
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## LIVE: one small unit, trained here

    A miniature run (a quarter of the benchmark budget, one net seed) on the d=3
    necklace under **iso15** noise — the regime where the entire variational
    program first earned its keep. Cached; delete the cache directory to retrain.
    """)
    return


@app.cell(hide_code=True)
def _(DEV, TDInfoNCE):
    from noise_benchmark import prepare_unit, pooled_dv


    def run_demo_unit(noise="iso15", nt=20, seed=0):
        caches = {}
        prep = prepare_unit(dict(profile="nk2", d=3, wseed=0, noise=noise),
                            caches, nt=nt)
        est = TDInfoNCE(gamma=prep["gamma_c"], z_dim=128, features="coords",
                        hidden=256, n_epochs=200, batch_size=4096,
                        n_candidates=511, lr=3e-4, holdout_frac=0.5,
                        lags_per_step=16, aug_scale=0.5, aug_nugget="auto",
                        device=DEV, seed=seed).fit(
            prep["traj_idx"], prep["n_pts"], X=prep["X"])
        field = pooled_dv(est, prep["groups"], rng=seed)
        return prep["kt_w"], field, float(est.cv_), float(est.jitter_)

    return (run_demo_unit,)


@app.cell(hide_code=True)
def demo_run(run_demo_unit):
    with mo.persistent_cache(name="td4_demo"):
        demo_kt, demo_field, demo_cv, demo_jit = run_demo_unit()
    return demo_cv, demo_field, demo_jit, demo_kt


@app.cell(hide_code=True)
def _(demo_cv, demo_field, demo_jit, demo_kt, np, plt):
    from scipy.stats import pearsonr

    _mm = np.isfinite(demo_field) & np.isfinite(demo_kt)
    _r = pearsonr(demo_field[_mm], demo_kt[_mm])[0]
    _f4, _a4 = plt.subplots(figsize=(4.6, 3.6), constrained_layout=True)
    _a4.scatter(demo_kt[_mm], demo_field[_mm], s=26, color="#2A6F97",
                alpha=0.85)
    _a4.set(xlabel="true scalar curvature R·s²",
            ylabel="td4 field (DV readout)",
            title=f"iso15, d=3, quarter budget: r = {_r:+.2f}\n"
                  f"(auto-rule chose jitter {demo_jit:.3f}, CV {demo_cv:.2f})")
    _f4
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## THE LEDGER: five channels, six noises (N-suite, nt=80)

    Two successor-entropy references (empirical-chain bins; kNN-graph resolvent),
    the FB-trained entropy (FBSENT), plain TD-InfoNCE, and td4's ancestor td_aug.
    The pattern that drove the campaign: *counting wins clean, learning wins
    noisy* — and the crossover sits exactly at the embedding-like regimes.
    """)
    return


@app.cell(hide_code=True)
def _(PD13, pd):
    nb = pd.read_csv(PD13 / "noise_bench_unified.csv")
    ledger = (nb.groupby(["noise", "d"])[
        ["r_sent_bin", "r_sent_knn", "r_fbsent", "r_td", "r_tdaug"]]
        .mean().round(2))
    ledger
    return


@app.cell(hide_code=True)
def _(PD13, pd, plt):
    i4 = pd.read_csv(PD13 / "iter4.csv")
    _sc = i4.groupby(["noise", "d"])["r_td4"].mean().unstack()
    _f5, _a5 = plt.subplots(figsize=(6.4, 3.6), constrained_layout=True)
    _im = _a5.imshow(_sc.values, cmap="RdYlGn", vmin=0.2, vmax=0.9)
    _a5.set_xticks(range(_sc.shape[1]), [f"d={c}" for c in _sc.columns])
    _a5.set_yticks(range(_sc.shape[0]), _sc.index)
    for _i in range(_sc.shape[0]):
        for _j in range(_sc.shape[1]):
            _v = _sc.values[_i, _j]
            _a5.text(_j, _i, f"{_v:.2f}" + (" ✓" if _v >= 0.70 else ""),
                     ha="center", va="center", fontsize=9,
                     fontweight="bold" if _v >= 0.70 else "normal")
    _a5.set_title("td4 scoreboard vs the 0.70 goal (nt=80, d=3–6)")
    _f5
    return (i4,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## WHAT 0.70 COSTS: the budget-scaling law

    Doubling trajectories at constant training compute. Unstructured noise obeys
    clean laws (+0.03–0.08 Pearson per doubling); the *structured* corruptions at
    high dimension are budget-flat — the estimator-innovation frontier.
    """)
    return


@app.cell(hide_code=True)
def _(PD13, i4, pd, plt):
    i5 = pd.read_csv(PD13 / "iter5.csv")
    _b80 = i4[["profile", "d", "wseed", "noise", "r_td4"]].rename(
        columns={"r_td4": "r_td5"}).assign(nt=80)
    _allr = pd.concat([_b80, i5[["profile", "d", "wseed", "noise", "nt",
                                 "r_td5"]]], ignore_index=True)
    _g = _allr.groupby(["noise", "d", "nt"])["r_td5"].mean().reset_index()
    _f6, _axs6 = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True,
                              constrained_layout=True)
    for _ax6, _nz in zip(_axs6, ("iso15", "hetero", "ar1")):
        for _d, _grp in _g[_g.noise == _nz].groupby("d"):
            _ax6.plot(_grp.nt, _grp.r_td5, "o-", label=f"d={_d}")
        _ax6.axhline(0.70, color="gray", ls="--", lw=0.8)
        _ax6.set(xscale="log", xlabel="trajectories nt", title=_nz)
        _ax6.legend(fontsize=7)
    _axs6[0].set_ylabel("field Pearson")
    _f6.suptitle("scaling toward 0.70: lawful (iso15) vs flat (ar1 at d≥4)")
    _f6
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## ROUND 6: breaking the flat cells

    Three innovations on the budget-flat cells, vs the td4 baseline. **Spectral
    coordinates** (learned Laplacian eigenfunctions as a denoised representation —
    the analytic-resolvent readout is unrescuable, a documented boundary result)
    own the mild-noise band; **temporal windows** own structured noise (built for
    ar1, won hetero); equalizing jitter died; **ar1 at d≥4 remains a
    three-null wall**.
    """)
    return


@app.cell(hide_code=True)
def _(PD13, i4, np, pd, plt):
    i6 = pd.read_csv(PD13 / "iter6.csv")
    _base6 = i4.groupby(["noise", "d"])["r_td4"].mean().rename("td4")
    _arms = i6.pivot_table(index=["noise", "d"], columns="arm",
                           values="r").join(_base6)
    _f7, _a7 = plt.subplots(figsize=(8.6, 3.4), constrained_layout=True)
    _lab = [f"{n} d{d}" for n, d in _arms.index]
    _xpos = np.arange(len(_arms))
    for _k, (_col, _colr) in enumerate(
            [("td4", "#888888"), ("spec", "#2A6F97"), ("win", "#C44536"),
             ("eq", "#CBB26A")]):
        _a7.bar(_xpos + (_k - 1.5) * 0.2, _arms[_col], width=0.19,
                label=_col, color=_colr)
    _a7.axhline(0.70, color="gray", ls="--", lw=0.8)
    _a7.set_xticks(_xpos, _lab, rotation=30, ha="right", fontsize=8)
    _a7.set_ylabel("field Pearson")
    _a7.set_title("round-6 arms on flat + sanity cells")
    _a7.legend(fontsize=8, ncol=4)
    _f7
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## VERDICTS, AND THE FENCE AT THE FRONTIER

    - **Goal met (≥0.70)**: clean d3/d4 (0.79 both; d4 via spectral coordinates),
      iso05 d3–5 (d4 = 0.79), clean d6 & iso05 d6 & iso15 d3 at nt ≤ 320.
    - **Moved off the flat**: hetero d5 → 0.61 (windows), hd64 d4 → 0.53.
    - **The open problem**: ar1 at d ≥ 4 — temporally correlated observation noise
      at high dimension has absorbed three targeted mechanisms without moving.
    - **The instrument**: td4 is self-calibrating end to end and, at d ≥ 5, beats
      even noise-free graph readouts — at high dimension there is currently no
      better instrument to be above.

    *Provenance: every number loads from `processed_data/iter*.csv`, produced by
    the committed army scripts (`iter{4,5,6}_benchmark.py`); the campaign
    narrative lives in the zettel
    [[20260404 Diffusion Curvature Revival with Trajectory Additions]].*
    """)
    return


if __name__ == "__main__":
    app.run()
