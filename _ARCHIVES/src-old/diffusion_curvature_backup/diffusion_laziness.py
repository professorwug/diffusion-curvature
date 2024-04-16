# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/library/Diffusion Laziness.ipynb.

# %% auto 0
__all__ = ['wasserstein_spread_of_diffusion', 'entropy_of_diffusion']

# %% ../../nbs/library/Diffusion Laziness.ipynb 3
import jax.numpy as jnp
from jax import jit

@jit
def wasserstein_spread_of_diffusion(
                D, # manifold geodesic distances
                Pt, # powered diffusion matrix/t-step ehat diffusions
                ):
        """
        Returns how "spread out" each diffusion is, with wasserstein distance
        Presumes that the manifold distances have been separately calculated
        """
        return jnp.sum(D * Pt, axis=-1)

# %% ../../nbs/library/Diffusion Laziness.ipynb 12
import jax.scipy
import jax.numpy as jnp
# def entropy_of_diffusion(Pt):
#         """
#         Returns the pointwise entropy of diffusion from the powered diffusion matrix in the input
#         Assumes that Pt sums to 1
#         """
#         Pt = (Pt + 1e-10) /(1 + 1e-10*Pt.shape[0]) # ensure, for differentiability, that there are no zeros in Pt, but that it still sums to 1.
#         entropy_elementwise = jax.scipy.special.entr(Pt)
#         entropy_of_rows = jnp.sum(entropy_elementwise, axis=-1)
#         # normalize so max value is 1
#         entropy_of_rows = entropy_of_rows / (-jnp.log(1/Pt.shape[0]))
#         return entropy_of_rows
def entropy_of_diffusion(Pt, epsilon=1e-5):
        """
        Returns the pointwise entropy of diffusion from the powered diffusion matrix in the input
        Assumes that Pt sums to 1
        """
        # Use only the elements of Pt that are greater than epsilon
        Pt = Pt * (Pt>epsilon)
        # Normalize Pt so that it sums to 1
        Pt = Pt / (jnp.sum(Pt, axis=-1) + 1e-12)
        # Pt = (Pt + 1e-10) /(1 + 1e-10*Pt.shape[0]) # ensure, for differentiability, that there are no zeros in Pt, but that it still sums to 1.
        entropy_elementwise = jax.scipy.special.entr(Pt)
        entropy_of_rows = jnp.sum(entropy_elementwise, axis=-1)
        # normalize so max value is 1
        # entropy_of_rows = entropy_of_rows / (-jnp.log(1/jnp.sum(Pt>epsilon, axis=-1)))
        return entropy_of_rows