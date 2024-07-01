import marimo

__generated_with = "0.6.14"
app = marimo.App()


@app.cell
def __():
    import diffusion_curvature
    from diffusion_curvature.data.toys import torus
    from diffusion_curvature.utils import plot_3d
    return diffusion_curvature, plot_3d, torus


@app.cell
def __():
    from diffusion_curvature.core import DiffusionCurvature
    return DiffusionCurvature,


@app.cell
def __(plot_3d, torus):
    X, ks = torus(2000)
    plot_3d(X,ks)
    return X, ks


@app.cell
def __(X):
    from diffusion_curvature.core import default_fixed_graph_former
    G = default_fixed_graph_former(X)
    return G, default_fixed_graph_former


@app.cell
def __(DiffusionCurvature, G, X):
    DC = DiffusionCurvature()
    diffusion_ks = DC.fit_transform(G = G, X = X, dim = 2)
    return DC, diffusion_ks


@app.cell
def __(X, diffusion_ks, plot_3d):
    plot_3d(X, diffusion_ks, colorbar=True)
    return


@app.cell
def __():
    return


if __name__ == "__main__":
    app.run()
