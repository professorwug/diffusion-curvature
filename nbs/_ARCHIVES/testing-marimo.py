import marimo

__generated_with = "0.9.20"
app = marimo.App(auto_download=["html"])


@app.cell
def __():
    from diffusion_curvature.datasets import torus
    from diffusion_curvature.core import DiffusionCurvature
    from diffusion_curvature.sadspheres import SadSpheres
    return DiffusionCurvature, SadSpheres, torus


app._unparsable_cell(
    r"""
    SS = SadSpheres(dimension=)
    """,
    name="__"
)


@app.cell
def __():
    from sklearn.datasets import make_moons
    import plotly.express as px

    def plot_overlapping_half_moons(n_samples: int = 100, noise: float = 0.1):
        """
        Generate a scatter plot of overlapping half moons.

        Parameters
        ----------
        n_samples : int
            The total number of points generated.
        noise : float
            Standard deviation of Gaussian noise added to the data.

        Returns
        -------
        None
        """
        try:
            X, y = make_moons(n_samples=n_samples, noise=noise)
            fig = px.scatter(x=X[:, 0], y=X[:, 1], color=y.astype(str), labels={'color': 'Class'})
            fig.update_layout(title='Overlapping Half Moons Scatter Plot', xaxis_title='X Coordinate', yaxis_title='Y Coordinate')
            fig.show()
        except Exception as e:
            print(f"Failed to generate scatter plot: {e}")

    plot_overlapping_half_moons()
    return make_moons, plot_overlapping_half_moons, px


if __name__ == "__main__":
    app.run()
