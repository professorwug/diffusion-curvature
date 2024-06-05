__all__ = ['enhanced_spectral_clustering']

from sklearn.cluster import KMeans
import jax.numpy as jnp
def enhanced_spectral_clustering(
        G, # PyGSP graph, or another with laplacian matrix L 
        ks, # unsigned magnitude of curvature
        dim, # intrinsic dimension of manifold (averaged across graph)
        num_clusters,
        curvature_weighting=1
        ):
    A = jnp.array(G.W.toarray())
    L = jnp.diag(jnp.sum(A,axis=1)) - A # D - A
    w, v = jnp.linalg.eigh(L) 
    X = v[:, 1:dim+1] # eigencoords
    # concatenate ks to X
    X = jnp.hstack((X, ks[:,None]))
    # normalize X by max min scaling
    X = (X - X.min(axis=0))/(X.max(axis=0) - X.min(axis=0))
    X = X.at[:,-1].set(curvature_weighting*X[:,-1])
    # perform knn clustering on X
    kmeans = KMeans(n_clusters=num_clusters).fit(X)
    return kmeans.labels_
    
