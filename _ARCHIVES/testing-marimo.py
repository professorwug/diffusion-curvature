import marimo

__generated_with = "0.1.79"
app = marimo.App()


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


if __name__ == "__main__":
    app.run()
