import marimo

__generated_with = "0.23.13"
app = marimo.App(width="medium")

with app.setup:
    import sys
    from pathlib import Path

    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    _here = Path(__file__).parent
    sys.path.insert(0, str(_here / "../11-successor-curvatures"))

    from diffusion_curvature.menagerie import (
        WarpedProduct,
        dumbbell_profile,
        necklace_profile,
        torus_flat,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # The Curvature Menagerie & the MDS-Strain Channel

    This essay introduces the two headline artifacts of the summer's curvature
    program: a **benchmark of manifolds with analytically certified curvature
    fields** (the *menagerie*), and the strongest estimator measured on it —
    the **MDS-strain channel**, a signed, natively-zeroed curvature field
    estimator that needs *no calibration object of any kind*.

    **THE PRINCIPLE THAT UNLOCKS BOTH.** Every estimator in our stack consumes
    a distance matrix, never an embedding. A benchmark manifold therefore needs
    only (i) an intrinsic sampling law and (ii) exact geodesic distances —
    which *warped products* supply in closed form, along with per-point
    analytic curvature. And an estimator needs only to interrogate that
    distance structure — which the MDS strain does through the oldest question
    in geometry processing: *does this patch embed flat?*
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## PART I: THE MENAGERIE, OR MANIFOLDS WITH BIRTH CERTIFICATES

    A **warped product** over the sphere carries the metric

    $$g = dr^2 + f(r)^2\, g_{S^{d-1}},$$

    one scalar *profile* $f$ sculpting the entire geometry. Its curvature is
    analytic in $f$ at every point:

    $$K_{\text{rad}} = -\frac{f''}{f}, \qquad
      K_{\text{tan}} = \frac{1 - f'^2}{f^2}, \qquad
      R = 2(d{-}1)K_{\text{rad}} + (d{-}1)(d{-}2)K_{\text{tan}}.$$

    Two profile families populate the battery:

    - **Dumbbells** (capped interval, topology $S^d$): two positively curved
      bulbs joined by a negatively curved neck — the canonical
      community-with-bottleneck geometry.
    - **Necklaces** (periodic profile, topology $S^1 \times S^{d-1}$):
      alternating pearls and necks, **boundary-free** — the honest negative
      reference that hyperbolic balls can never be (their rim-heavy sampling
      is all boundary).

    Geodesics reduce to a 2-D surface of revolution regardless of $d$
    (validated at 0.6% mean error against exact sphere geometry), so exact
    distance matrices cost seconds. Every sampled point arrives with its true
    scalar curvature — a birth certificate the old function-graph benchmarks
    never issued.
    """)
    return


@app.function
def revolution_embedding(profile_fn, L, n_r=160, n_theta=80, periodic=False):
    """Isometric (where |f'|<=1) surface-of-revolution embedding of a d=2
    warped product: x = f cos(theta), y = f sin(theta), z = integral of
    sqrt(max(0, 1 - f'^2)). Returns meshgrid arrays (X, Y, Z) and the scalar
    curvature field R(r) on the mesh for coloring."""
    r = np.linspace(0, L, n_r) if not periodic else np.linspace(
        0, L, n_r, endpoint=False)
    f = profile_fn(r)
    fp = np.gradient(f, r)
    fpp = np.gradient(fp, r)
    dz = np.sqrt(np.maximum(0.0, 1.0 - fp**2))
    z = np.concatenate([[0], np.cumsum(0.5 * (dz[1:] + dz[:-1]) * np.diff(r))])
    theta = np.linspace(0, 2 * np.pi, n_theta)
    F, TH = np.meshgrid(f, theta, indexing="ij")
    Z = np.meshgrid(z, theta, indexing="ij")[0]
    X, Y = F * np.cos(TH), F * np.sin(TH)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(f > 1e-9, -fpp / np.maximum(f, 1e-9), 0.0)
        Kt = np.where(f > 1e-9, (1 - fp**2) / np.maximum(f, 1e-9) ** 2, 0.0)
    R = 2 * K + 0 * Kt  # d=2: R = 2*K_rad (K_tan term has (d-2)=0 factor)
    Rm = np.meshgrid(R, theta, indexing="ij")[0]
    return X, Y, Z, Rm


@app.cell
def _():
    _f_d, _L_d = dumbbell_profile(beta=0.65)
    _Xd, _Yd, _Zd, _Rd = revolution_embedding(_f_d, _L_d)
    _f_n, _L_n = necklace_profile(a=1.0, b=0.55, k=2)
    _Xn, _Yn, _Zn, _Rn = revolution_embedding(_f_n, _L_n, periodic=True)

    _common = dict(
        colorscale="RdBu_r", cmid=0.0,
        colorbar=dict(title="scalar curvature R", len=0.7),
        showscale=True,
    )
    fig_manifolds = go.Figure()
    fig_manifolds.add_trace(go.Surface(
        x=_Xd, y=_Yd, z=_Zd, surfacecolor=_Rd, name="dumbbell", **_common))
    fig_manifolds.add_trace(go.Surface(
        x=_Xn + 3.2, y=_Yn, z=_Zn - _Zn.mean() + _Zd.mean(),
        surfacecolor=_Rn, showscale=False, colorscale="RdBu_r", cmid=0.0,
        name="necklace"))
    fig_manifolds.update_layout(
        title="The menagerie's two profile families (d=2 sections), colored "
              "by analytic scalar curvature: red = positive bulbs/pearls, "
              "blue = negative necks",
        scene=dict(aspectmode="data",
                   xaxis_visible=False, yaxis_visible=False,
                   zaxis_visible=False),
        margin=dict(l=0, r=0, t=60, b=0), height=520,
    )
    fig_manifolds
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Above: the **dumbbell** (left) and a two-pearl **necklace** (right) as
    honest surfaces of revolution, colored by their *certified* curvature
    field. In the battery these live at $d = 3\ldots6$ with $n = 1400$ points,
    exact pairwise geodesics, multiplicative distance-noise variants, and
    unit-median-distance normalization ($k_s \mapsto k_s \cdot s^2$: curvature
    in data-scale units, the only estimable kind).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## PART II: THE MDS-STRAIN CHANNEL, OR ASKING A PATCH TO LIE FLAT

    Take a small geodesic ball of $m$ points around an evaluation point,
    with squared distances $D^2$. Classical MDS double-centers it,

    $$B = -\tfrac{1}{2} J D^2 J, \qquad J = I - \tfrac{1}{m}\mathbf{1}\mathbf{1}^{\!\top},$$

    and asks: *what Gram matrix generated these distances?* In **flat** space
    the answer has rank exactly $d$ — every eigenvalue beyond the top $d$ is
    **identically zero**. Curvature breaks this: the residual spectrum
    acquires a *signed* structure (positive curvature compresses cross-patch
    distances, negative expands them, and the tail eigenvalues shift in
    opposite characteristic ways — derived empirically on
    sphere/torus/hyperbolic triads and frozen before any benchmark was run).

    The channel is simply the normalized signed residual

    $$\text{score}(p) = -\;\frac{\sum_{i > d} \lambda_i}{\sum_{i \le d} \lambda_i},$$

    higher = more positively curved, with three properties no prior channel
    combined:

    1. **An analytic native zero** — flat patches give machine-zero residual
       (1e-16), so `score = 0` *is* the sign boundary. No flat packs, no
       comparison spaces, no calibration.
    2. **A different failure algebra** — every other channel reads diffusion
       (and shares mixing/density/spread failure modes); this one reads
       embedding rigidity. It is the first channel to crack the equal-density
       homogeneous exam (sign = 1.00 at every dimension) where all diffusion
       channels saturate.
    3. **Cheap** — one small eigendecomposition per patch; the dimension can
       even be estimated from the eigengap.
    """)
    return


@app.function
def mds_strain_score(D, eval_idx, d, m=60):
    """The V4 MDS-strain channel (headline configuration).

    Args:
        D: (n, n) geodesic distance matrix.
        eval_idx: indices of evaluation points.
        d: intrinsic dimension (or an eigengap estimate).
        m: patch size in points (60 = validated default; larger sharpens
           high-dim sign but risks wrap-around on small manifolds).

    Returns:
        (len(eval_idx),) signed scores; >0 = positive curvature, exact 0 on
        flat patches. Field r 0.91-0.98 on dumbbells d3-6; sign-at-zero
        0.85-1.00 across the menagerie ladder.
    """
    scores = np.empty(len(eval_idx))
    for q, p in enumerate(np.asarray(eval_idx, dtype=int)):
        patch = np.argsort(D[p])[:m]
        D2 = D[np.ix_(patch, patch)] ** 2
        J = np.eye(m) - np.ones((m, m)) / m
        B = -0.5 * J @ D2 @ J
        eig = np.sort(np.linalg.eigvalsh(B))[::-1]
        top = eig[:d].sum()
        resid = eig[d:].sum()
        scores[q] = -resid / max(top, 1e-30)
    return scores


@app.cell
def _():
    def _build_dumbbell_arrays():
        _f3, _L3 = dumbbell_profile(beta=0.8)
        _m = WarpedProduct(_f3, _L3, d=3).sample(1100, rng=0)
        return _m["D"], _m["ks_field"], _m["r"]

    with mo.persistent_cache("dumbbell_d3_demo"):
        _D3, _ks3, _r3 = _build_dumbbell_arrays()

    _s = np.median(_D3[np.triu_indices_from(_D3, 1)])
    D_demo, ks_demo, r_demo = _D3 / _s, _ks3 * _s**2, _r3

    _order = np.argsort(ks_demo)
    eval_demo = _order[(np.linspace(0.02, 0.98, 120) * (len(ks_demo) - 1)).astype(int)]
    score_demo = mds_strain_score(D_demo, eval_demo, d=3, m=60)
    field_r = float(np.corrcoef(score_demo, ks_demo[eval_demo])[0, 1])
    _pos = ks_demo[eval_demo] > 0
    _neg = ks_demo[eval_demo] < 0
    sign_acc = 0.5 * ((score_demo[_pos] > 0).mean() + (score_demo[_neg] < 0).mean())
    return eval_demo, field_r, ks_demo, r_demo, score_demo, sign_acc


@app.cell
def _(eval_demo, field_r, ks_demo, score_demo, sign_acc):
    _fig = go.Figure()
    _fig.add_trace(go.Scatter(
        x=ks_demo[eval_demo], y=score_demo, mode="markers",
        marker=dict(color=np.sign(ks_demo[eval_demo]),
                    colorscale="RdBu_r", size=7, line=dict(width=0.5)),
        name="eval points"))
    _fig.add_hline(y=0, line_dash="dot")
    _fig.add_vline(x=0, line_dash="dot")
    _fig.update_layout(
        title=(f"MDS-strain on a live d=3 dumbbell (n=1100): "
               f"field r = {field_r:+.2f}, balanced sign at the native zero "
               f"= {sign_acc:.2f} — the dotted crosshair IS the classifier"),
        xaxis_title="certified scalar curvature (data-scale units)",
        yaxis_title="MDS-strain score",
        height=440, margin=dict(l=0, r=0, t=60, b=0))
    _fig
    return


@app.cell
def _(eval_demo, ks_demo, r_demo, score_demo):
    _f_d2, _L_d2 = dumbbell_profile(beta=0.8)
    _X, _Y, _Z, _ = revolution_embedding(_f_d2, _L_d2)
    _fp = np.gradient(_f_d2(np.linspace(0, _L_d2, 400)),
                      np.linspace(0, _L_d2, 400))
    _rg = np.linspace(0, _L_d2, 400)
    _dz = np.sqrt(np.maximum(0.0, 1.0 - _fp**2))
    _zg = np.concatenate([[0], np.cumsum(0.5 * (_dz[1:] + _dz[:-1])
                                         * np.diff(_rg))])

    _rr = r_demo[eval_demo]
    _th = np.random.default_rng(1).uniform(0, 2 * np.pi, len(_rr))
    _fr = _f_d2(_rr)
    _zz = np.interp(_rr, _rg, _zg)

    fig_field = go.Figure()
    fig_field.add_trace(go.Surface(
        x=_X, y=_Y, z=_Z, opacity=0.25, showscale=False,
        colorscale=[[0, "#dddddd"], [1, "#dddddd"]]))
    for _vals, _name, _dx in [(ks_demo[eval_demo], "certified K", 0.0),
                              (score_demo, "MDS-strain estimate", 3.0)]:
        fig_field.add_trace(go.Scatter3d(
            x=_fr * np.cos(_th) + _dx, y=_fr * np.sin(_th), z=_zz,
            mode="markers",
            marker=dict(size=4, color=np.sign(_vals), colorscale="RdBu_r",
                        cmin=-1, cmax=1),
            name=_name))
    fig_field.update_layout(
        title="Sign fields on the dumbbell (schematic d=2 section): "
              "certified (left) vs MDS-strain estimate (right) — "
              "red positive bulbs, blue negative neck",
        scene=dict(aspectmode="data", xaxis_visible=False,
                   yaxis_visible=False, zaxis_visible=False),
        height=480, margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(orientation="h"))
    fig_field
    return


@app.cell
def _():
    with mo.persistent_cache("flat_null_demo"):
        _t = torus_flat(900, 3, rng=0)
        _Dt = _t["D"] / np.median(_t["D"][np.triu_indices_from(_t["D"], 1)])
        _idx = np.random.default_rng(0).choice(900, 120, replace=False)
        flat_scores = mds_strain_score(_Dt, _idx, d=3, m=60)

    _fig = go.Figure()
    _fig.add_trace(go.Histogram(x=flat_scores, nbinsx=40,
                                name="flat torus T\u00b3"))
    _fig.update_layout(
        title=(f"The native zero, demonstrated: MDS-strain on a certified-"
               f"flat T\u00b3 \u2014 median |score| = {np.median(np.abs(flat_scores)):.2e} "
               "(the rank-d limit is analytically exact; the residual here "
               "is torus min-image wrap at this modest n \u2014 see the ladder "
               "for the 1e-16 nulls at n=1400)"),
        xaxis_title="MDS-strain score", height=320,
        margin=dict(l=0, r=0, t=60, b=0))
    _fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## THE LEDGER, FOR THE RECORD

    Benchmark numbers from the frozen ladder (`v4_mds_*` in
    `processed_data/`; field Pearson | balanced sign at the native zero,
    noiseless / 15% distance noise):

    | dim | dumbbell | necklace |
    |---|---|---|
    | 3 | 0.98/0.99 · 0.95/0.99 | 0.97/0.96 · 0.95/0.90 |
    | 4 | 0.96/0.99 · 0.92/0.99 | 0.96/0.99 · 0.93/0.82 |
    | 5 | 0.91/0.88 · 0.89/0.74 | −0.05/0.89 · 0.01/0.79 |
    | 6 | 0.92/1.00 · 0.82/1.00 | −0.20/0.85 · −0.16/0.73 |

    The one honest hole: at high-dim necklaces the *within-sign magnitude
    ordering* inverts (field r goes negative) even as the sign holds — the
    open problem the channel bequeaths to its successors. Robust to Dijkstra
    graph geodesics (real data graphs, not just certified distances) and to
    the eigengap dimension estimate.

    *Provenance: developed 2026-07-10 in the V-series (flattening-flow
    program); see the revival zettel for the full falsification trail that
    led here.*
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## PART III: THE SUITE ASSEMBLED, OR MANY INSTRUMENTS, ONE VERDICT

    No single channel covers everything — each has a mapped regime. The
    **channel suite** fuses seven of them with a small learned integrator:

    | channel | family | native regime |
    |---|---|---|
    | `v4_m60`, `v4_m120` | MDS strain (embedding rigidity) | strong-curvature intrinsic fields; homogeneous manifolds |
    | `v3_defect` | second-moment transport defect | strong-curvature fields (corroborant) |
    | `sent` | resolvent successor entropy | collar-type geometry, magnitude ruler |
    | `ent_cak` | scale-tuned diffusion entropy | **weak-curvature embedded clouds** (colosseum) |
    | `frac` | diffusing-edge fraction | bottleneck geometry, sparse regimes |
    | `kappa` | Diffusion ORC ($W_1$ contraction) | point estimates, positive side |

    **Integration procedure.** Per evaluation point: extract all channels from
    the distance matrix alone, plus four context features (eigengap dimension
    estimate, measure spread, kNN radius, patch wrap fraction); orient so
    higher = more positive $K$; z-score against a **regime-matched flat
    reference** (flat tori for intrinsic distance data; embedded flat planes
    for ambient pointclouds — mismatching this is *the* transfer failure mode);
    feed a depth-4 gradient-boosted pair of heads (sign classifier + magnitude
    regressor) trained on ~72 **randomized-profile** warped manifolds plus 33
    homogeneous constant-$K$ manifolds (spheres, hyperbolic balls, tori — the
    augmentation that closed the out-of-distribution gap). The colosseum is
    never trained on: it is a pure transfer test.
    """)
    return


@app.cell(hide_code=True)
def _():
    _pd = __import__("pandas")
    _base = mo.notebook_dir() / "processed_data"
    _e1 = _pd.read_csv(_base / "suite_e1_summary_aug.csv")
    _e2 = _pd.read_csv(_base / "suite_e2_summary_aug.csv")
    _e3 = _pd.read_csv(_base / "suite_e3_summary_aug.csv")
    _e4 = _pd.read_csv(_base / "suite_e4_summary_aug.csv")

    _rows = []
    for _d in sorted(_e1.dim.unique()):
        _g1 = _e1[(_e1.dim == _d) & (_e1.noise == 0.0)]
        for _kind, _gk in _g1.groupby("kind"):
            _rows.append(dict(battery=f"menagerie {_kind} (held-out)", dim=_d,
                              metric="field r | sign",
                              integrator=f"{_gk.r_integ.mean():.2f} | {_gk.sign_integ.mean():.2f}",
                              best_single=f"V4: {_gk.r_v4_m60.mean():.2f} | {_gk.sign_v4_m60.mean():.2f}"))
    for _d in sorted(_e2.dim.unique()):
        _g = _e2[_e2.dim == _d]
        _rows.append(dict(battery="tier1-v2 homogeneous", dim=_d,
                          metric="cross-manifold sign",
                          integrator=f"{_g.sign_integ.mean():.2f}",
                          best_single=f"V4: {_g.sign_v4_m60.mean():.2f}"))
    for _d in sorted(_e3.dim.unique()):
        _g = _e3[_e3.dim == _d]
        _rows.append(dict(battery="colosseum (pure transfer)", dim=_d,
                          metric="field r | sign",
                          integrator=f"{_g.r_integ.mean():.2f} | {_g.sign_integ.mean():.2f}",
                          best_single=f"ent_cak: {_g.r_ent_cak.mean():.2f} | {_g.sign_ent_cak.mean():.2f}"))
    for _d in sorted(_e4.dim.unique()):
        _g = _e4[_e4.dim == _d]
        _rows.append(dict(battery="SadSpheres", dim=_d, metric="AUC | sign",
                          integrator=f"{_g.auc_integ.mean():.2f} | {_g.sign_integ.mean():.2f}",
                          best_single=f"V4: {_g.auc_v4_m60.mean():.2f}"))
    coverage_table = _pd.DataFrame(_rows)
    mo.ui.table(coverage_table, page_size=25, label="Full coverage: integrator vs best single channel (augmented model; noiseless cells)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **READING THE COVERAGE TABLE.** Four verdicts, one per battery:

    1. **Menagerie held-out families** — the integrator *ties V4 where V4 is
       perfect and wins where every single channel fails*: at high-dim
       necklaces (d=5–6) it holds sign 0.91–0.95 and positive field r where
       V4-alone drops to 0.55–0.64 and all channels (and the isotonic
       baseline) go *negative*. Multivariate fusion earns its keep exactly at
       the hard cells.
    2. **Equal-density homogeneous exam** — V4 alone scores sign = 1.00 at
       every dimension, the first channel ever to crack this test (all
       diffusion channels saturate to chance); the integrator matches it after
       homogeneous augmentation (0.91–1.00; without augmentation it inverted —
       a training-distribution gap, fixed for free).
    3. **Colosseum (pure transfer)** — the transfer is carried by the
       *diffusion* channels: `ent_cak` alone reaches r = 0.55–0.77 per dim,
       matching the hand-built v6 composite, while V3/V4 sit at the floor
       (colosseum |K| ~ 1–6 lies below their signal thresholds). Sign
       transfers once the reference matches the regime: swapping the
       flat-torus reference for an **embedded flat-plane** reference restores
       balanced sign from chance to **0.72/0.76/0.70 at d=3–5 (mean 0.67 ≈
       v6's 0.68, beating v6 at its weakest cells d3–4)** — confirming the
       failure was zero-calibration, never the channels
       (`suite_e3_rescore.csv`). One honest tension remains: homogeneous
       augmentation helps the homogeneous exam but hurts colosseum (it shifts
       weight onto V4, which is floor-bound there) — reference *and* training
       distribution must both match the data regime; a single
       regime-spanning model is the open refinement.
    4. **SadSpheres** — AUC 1.00 for V4, sent, and ent_cak at every dimension;
       the integrator 0.89–1.00, with sign-at-zero rising to
       **0.70/1.00/1.00/0.93/1.00** under the embedded reference.

    **THE SENTENCE THE WHOLE PROGRAM EARNED.** Curvature sign from sampled
    geometry is solved by *static, pooled, natively-zeroed channels with
    regime-matched references and learned fusion* — not by flows, warps, or
    dynamics, each of which was tried and fell to an identified mechanism
    (noise compounding, potential overwrite, seeding circularity, entropy
    production, re-derivation). The dead ends are documented in the revival
    zettel; the survivors are in this notebook.
    """)
    return


if __name__ == "__main__":
    app.run()
