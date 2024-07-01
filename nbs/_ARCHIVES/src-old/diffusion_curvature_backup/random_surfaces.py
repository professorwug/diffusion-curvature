# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/library/Random Surfaces.ipynb.

# %% auto 0
__all__ = ['random_polynomial', 'random_surface', 'manifold_density', 'max_value', 'rejection_sample_from_surface',
           'second_fundamental_form', 'riemannian_curvature_tensor', 'scalar_curvature_at_origin',
           'samples_from_random_surface']

# %% ../../nbs/library/Random Surfaces.ipynb 4
import sympy as sp
import numpy as np
import itertools

def random_polynomial(
        vars, # variables to construct polynomial from
        degree = 2, # maximum degree of terms
):
    num_variables = len(vars)
    terms = []
    for d in range(1, degree + 1):
        for indices in itertools.combinations_with_replacement(range(num_variables), d):
            terms.append(np.prod([vars[i] for i in indices]))
    coeffs = np.random.normal(size = len(terms))
    return sum([coeffs[i] * terms[i] for i in range(len(terms))])



# %% ../../nbs/library/Random Surfaces.ipynb 7
def random_surface(
        d:int, # intrinsic dimension, number of tangent coordinates
        N:int, # ambient dimension,
        degree = 2, # maximum degree of terms
):
    # To convert our random polynomials into a surface, this function constructs a 1d sympy matrix $f$ where for the first d entries, $f_1 = x_i$. Then, for the remaining $N - d$ entries, $f_i$ is assigned to a random polynomial computed with the random_polynomial function.
    # The output is a sympy matrix of size $N \times 1$.
    vars = sp.symbols('x0:%d' % d)
    f = sp.Matrix([*vars])
    for i in range(d, N):
        f = f.row_insert(i, sp.Matrix([random_polynomial(vars, degree)]))
    return f, vars

# %% ../../nbs/library/Random Surfaces.ipynb 16
from tqdm.auto import tqdm
def manifold_density(f, variables):
    G = sp.Matrix.zeros(len(variables), len(variables))
    for i, x1 in enumerate(variables):
        for j, x2 in enumerate(variables):
            G[i,j] = (sp.diff(f, x1).T  * sp.diff(f, x2))[0]
    return sp.sqrt(G.det(method="lu"))

# %% ../../nbs/library/Random Surfaces.ipynb 19
import numpy as np
import scipy.optimize as opt
import sympy as sp

def max_value(expr, bounds):
    # Convert sympy expression to numpy function
    vars = list(expr.free_symbols)
    expr_neg = -1*expr
    neg_func = sp.lambdify([vars], expr_neg, 'numpy')

    # Minimize negative of function over range [-1, 1]
    bounds = [bounds for _ in vars]
    res = opt.minimize(neg_func, np.zeros(len(vars)), bounds=bounds)
    
    # Return maximum value
    return -res.fun


# %% ../../nbs/library/Random Surfaces.ipynb 21
def rejection_sample_from_surface(
        F, # a sympy matrix of size $N \times 1$ representing a surface
        n_points, # number of points to sample
        bounds=[-1,1], # bounds for each variable
        batch_size=1024, # number of points to test sampling at a time
        verbose=False,
):
    if verbose: print("Hey, just woke up")
    vars = list(F.free_symbols)
    f = manifold_density(F, vars)
    g = 1/((bounds[1]-bounds[0])**len(vars)) # uniform density on [-1, 1] for each variable
    M = max_value(f/g, bounds=bounds) # M >= f/g for all x
    if verbose: print("Computed f, M, g")
    bouncer = (f / (M * g)) #.simplify()
    # print(bouncer)
    points = []
    # convert f to numpy
    F_np = sp.lambdify([vars], F, 'numpy')
    bouncer_np = sp.lambdify([vars], bouncer, 'numpy')
    # add the origin as first point
    points.append(F_np(np.zeros(len(vars))))
    if verbose: print("Computed bouncer np")
    while len(points) < n_points:

        euc_coords = np.random.uniform(bounds[0], bounds[1], (batch_size,len(vars)))
        x = np.array(list(map(F_np,euc_coords)))
        if verbose: print("computed sample candidates")
        u = np.random.uniform(0, 1, batch_size)
        # print(u)
        # compute mask of points that pass the bouncer
        bouncer_results = np.array(list(map(bouncer_np,euc_coords)))
        if verbose: print("computed bouncer results")
        mask = u < bouncer_results
        points.extend(x[mask])
        if verbose:
            print(f"Points added {np.sum(mask)} for a total of {len(points)}")
        # if u < bouncer_np(euc_coords):
        #     points.extend(x)
    if len(points) > n_points:
        points = points[:n_points]
    return np.squeeze(np.array(points))

# %% ../../nbs/library/Random Surfaces.ipynb 25
def second_fundamental_form(F):
    vars = list(F.free_symbols)
    d = len(vars)
    N = len(F)
    H = np.zeros([d,d,N-d]) # d x d x (N-d); the last dimension holds the fk's
    for i, x1 in enumerate(vars):
        for j, x2 in enumerate(vars):
            H[j,i] = np.squeeze(sp.diff((sp.diff(F, x1)),x2).subs([(v,0) for v in vars]))[-(N-d):] # evaluate at the origin
    return H

# %% ../../nbs/library/Random Surfaces.ipynb 27
def riemannian_curvature_tensor(F):
    H = second_fundamental_form(F)
    # print(H)
    N = H.shape[-1]
    d = H.shape[0]
    R = np.zeros((d,d,d,d))
    
    # Edward's method
    # g = np.eye(N)
    # hprod1 = np.einsum('jka,ilb->ijklab',H,H)
    # part1 = np.einsum('ijklab,ab->ijkl',hprod1,g)

    # hprod2 = np.einsum('jib,kla->ijklab',H,H)
    # part2 = np.einsum('ijklab,ab->ijkl',hprod2,g)

    # R = part1 - part2

    # Idx flip suggested by Copilot's autocomplete; it must have seen those flipped idxs in training
    # for i in range(d):
    #     for j in range(d):
    #         for k in range(d):
    #             for l in range(d):
    #                 # We sum over the number of normal functions N-d, representing this value for each fk
    #                 R[i,j,k,l] = np.sum(H[i, k] * H[j, l] - H[i, l] * H[j, k]) # np.sum(H[j,k]*H[i,l] - H[j,i]*H[k,l])
    
    # Sritharan's Method, adapted from [curvature_expression.m · master · hormozlab / ManifoldCurvature · GitLab](https://gitlab.com/hormozlab/ManifoldCurvature/-/blob/master/curvature_expression.m?ref_type=heads)
    # The placement of idxs agrees with the second method, but the values are different
    H2 = np.outer(H.flatten(), H.flatten()).reshape(H.shape + H.shape)
    HC = sum(H2[:,:,i,:,:,i] for i in range(H.shape[-1]))
    ida = [0,2,1,3]
    idb = [2,1,0,3]
    HC_a = np.transpose(HC, ida)
    HC_b = np.transpose(HC, idb)
    R = HC_a - HC_b # Gauss's formula for R_ijkl
    
    return R

# %% ../../nbs/library/Random Surfaces.ipynb 28
def scalar_curvature_at_origin(F):
    R = riemannian_curvature_tensor(F)
    Ricc = sum([R[:,i,:,i] for i in range(R.shape[-1])])
    S = sum([Ricc[i,i] for i in range(Ricc.shape[-1])])
    # S = 0
    # for i in range(len(R)):
    #     for j in range(len(R)):
    #         S += R[i,j,i,j]
    return S

# %% ../../nbs/library/Random Surfaces.ipynb 31
def samples_from_random_surface(
        n_samples, # number of samples to generate
        d, # intrinsic dimension
        N, # ambient dimension
        degree = 2, # maximum degree of terms
        noise_level = 0, # standard deviation of noise to add to samples
        verbose=False,
):
    F, vars = random_surface(d, N, degree)
    X = rejection_sample_from_surface(F, n_samples, verbose=verbose)
    if noise_level > 0:
        X += np.random.normal(scale = noise_level, size = X.shape)
    k = scalar_curvature_at_origin(F)
    return X, k
