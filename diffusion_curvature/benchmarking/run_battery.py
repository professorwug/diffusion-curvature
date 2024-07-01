# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/4a-curvature-colosseum-with-diffusion-curvature.ipynb.

# %% auto 0
__all__ = ['compute_curvature_on_battery', 'compute_correlations', 'result_table', 'compute_sign_score']

# %% ../../nbs/4a-curvature-colosseum-with-diffusion-curvature.ipynb 4
# from fastprogress.fastprogress import master_bar, progress_bar
# Iterate through the battery and compute the diffusion curvature
# store it in a new dictionary
def compute_curvature_on_battery(
        curvature_function, # fn that, given X and dim, returns the curvature of the first point
        CC, # the battery dictionary
        restrict_to_first_n_dims = None,
        ):
    computed_curvature = {}
    dimensions = CC['dims']
    if restrict_to_first_n_dims:
        dimensions = dimensions[:restrict_to_first_n_dims]
    for d in tqdm(CC['dims'], desc="intrinsic dimensions"):
        computed_curvature[d] = {}
        for c in tqdm(CC[d]['codims'], leave=False, desc='codimensions'):
            computed_curvature[d][c] = {}
            for noise_level in tqdm(CC[d][c]['noise_levels'], leave=False, desc="Noise Levels"):
                computed_curvature[d][c][noise_level] = {}
                computed_curvature[d][c][noise_level]['k'] = []
                # apply curvature function in parallel
                # us = [(CC[d][c][noise_level]['Xs'][i], d) for i in range(len(CC[d][c][noise_level]['Xs']))]
                # ks = parallel(curvature_function, us, n_workers=25)
                # computed_curvature[d][c][noise_level]['k'] = ks
                for i in trange(len(CC[d][c][noise_level]['Xs']), leave=False, desc="Samples"):
                    X = CC[d][c][noise_level]['Xs'][i]
                    k = curvature_function(X, d)
                    computed_curvature[d][c][noise_level]['k'].append(k)
    return computed_curvature

# %% ../../nbs/4a-curvature-colosseum-with-diffusion-curvature.ipynb 9
# compute the pearson correlations between the computed curvature and the true curvature
def compute_correlations(
        computed_curvature, # the computed curvature
        CC, # the battery dictionary
        ):
    correlations = {}
    for d in tqdm(CC['dims'], desc="intrinsic dimensions"):
        correlations[d] = {}
        for c in tqdm(CC[d]['codims'], leave=False, desc='codimensions'):
            correlations[d][c] = {}
            for noise_level in tqdm(CC[d][c]['noise_levels'], leave=False, desc="Noise Levels"):
                correlations[d][c][noise_level] = {}
                correlations[d][c][noise_level]['r'] = []
                correlations[d][c][noise_level]['p'] = []
                # for i in trange(len(CC[d][c][noise_level]['Xs']), leave=False, desc="Samples"):
                k = computed_curvature[d][c][noise_level]['k']
                k_true = CC[d][c][noise_level]['k']
                r, p = pearsonr(k, k_true)
                correlations[d][c][noise_level]['r'] = r
                correlations[d][c][noise_level]['p'] = p
    return correlations

# %% ../../nbs/4a-curvature-colosseum-with-diffusion-curvature.ipynb 12
# Make a latex table of the correlations, both r and p values, with dimension in the rows and noise level in the columns
def result_table(
        correlations, # dictionary of correlations
        c:int, # codimension
        style = 'fancy_grid',
        keys = ['r','p']
        ):
    noise_levels = correlations[correlations.keys().__iter__().__next__()][c].keys()
    print("Codimension = ",c)
    table = tabulate(
        [[d] + [f"{correlations[d][c][noise_level][keys[0]]:.{3}f}{'/' + str(correlations[d][c][noise_level][keys[1]])[:4] if len(keys) > 1 else ''}" for noise_level in noise_levels] for d in correlations.keys()],
        headers=['dim'] + [f"Noise = {nl}" for nl in noise_levels],
        tablefmt=style, #latex_raw
        floatfmt=".2f",
        )
    print(table)
    return table

# %% ../../nbs/4a-curvature-colosseum-with-diffusion-curvature.ipynb 18
def compute_sign_score(
        computed_curvature, # the computed curvature
        CC, # the battery dictionary
        ):
    correlations = {}
    for d in tqdm(CC['dims'], desc="intrinsic dimensions"):
        correlations[d] = {}
        for c in CC[d]['codims']:
            correlations[d][c] = {}
            for noise_level in tqdm(CC[d][c]['noise_levels'], leave=False, desc="Noise Levels"):
                correlations[d][c][noise_level] = {}
                correlations[d][c][noise_level]['r'] = []
                correlations[d][c][noise_level]['p'] = []
                # for i in trange(len(CC[d][c][noise_level]['Xs']), leave=False, desc="Samples"):
                k = computed_curvature[d][c][noise_level]['k']
                k_true = CC[d][c][noise_level]['k']
                # measure the 'classification accuracy' of the signs.
                class_acc = np.sum(
                    (np.sign(k) == np.sign(k_true)).astype(int)
                ) / len(k)
                correlations[d][c][noise_level]['accuracy'] = class_acc
    return correlations
