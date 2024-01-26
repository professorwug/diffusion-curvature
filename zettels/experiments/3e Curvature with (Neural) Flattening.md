# 3e Curvature Via (Neural) Flattening

    [gpu(id=0)]

The bugbear of diffusion curvature, as it now stands, is that the local
sampling of each neighborhood is (especially in high dimensions) full of
pockmarks and holes that interfere with the curvature measurement.

One possible solution is to create a comparison space that have the same
deformities. This notebook tests two methods of doing this:

1.  Just taking a PCA projection of the data into the (known) intrinsic
    dimension, then building a graph in the euclidean PCA plane, and
    using this as the comparison graph.
2.  Performing “Neural Flattening” on the PCA’d space, to force points
    to migrate out of areas with high density into a more uniform
    distribution.

# The Normal Way

First, we create a dataset of saddles and take the diffusion curvature
on them in the normal way, with the mean precomputed comparison space.

``` python
import graphtools
from diffusion_curvature.core import DiffusionCurvature
from diffusion_curvature.datasets import rejection_sample_from_saddle
```

``` python
ks_dc = []
dim = 2
samplings = [200]*100
Xs_sampled = []
for n_points in tqdm(samplings):
    X, k = rejection_sample_from_saddle(n_points, dim)
    Xs_sampled.append(X)
    # Compute Diffusion Curvature
    G = graphtools.Graph(X, anisotropy=1, knn=5, decay=None).to_pygsp()
    DC = DiffusionCurvature(
        laziness_method="Entropic",
        flattening_method="Mean Fixed",
        comparison_method="Subtraction",
        points_per_cluster=None, # construct separate comparison spaces around each point
        comparison_space_size_factor=1
    )
    ks = DC.curvature(G, t=25, dim=dim, knn=5, idx=0)
    ks_dc.append(ks)
# plot a histogram of the diffusion curvatures
plt.hist(ks_dc, bins=20)
```

      0%|          | 0/100 [00:00<?, ?it/s]

    2023-12-08 15:24:09.030037: W external/xla/xla/stream_executor/gpu/asm_compiler.cc:231] Falling back to the CUDA driver for PTX compilation; ptxas does not support CC 8.9
    2023-12-08 15:24:09.030055: W external/xla/xla/stream_executor/gpu/asm_compiler.cc:234] Used ptxas at ptxas
    2023-12-08 15:24:09.030122: W external/xla/xla/stream_executor/gpu/redzone_allocator.cc:322] UNIMPLEMENTED: ptxas ptxas too old. Falling back to the driver to compile.
    Relying on driver to perform ptx compilation. 
    Modify $PATH to customize ptxas location.
    This message will be only logged once.
    2023-12-08 15:24:09.032957: W external/xla/xla/service/gpu/buffer_comparator.cc:641] UNIMPLEMENTED: ptxas ptxas too old. Falling back to the driver to compile.
    Relying on driver to perform ptx compilation. 
    Setting XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/cuda  or modifying $PATH can be used to set the location of ptxas
    This message will only be logged once.
    2023-12-08 15:24:09.063303: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:548] failed to load PTX text as a module: CUDA_ERROR_INVALID_IMAGE: device kernel image is invalid
    2023-12-08 15:24:09.063327: E external/xla/xla/stream_executor/cuda/cuda_driver.cc:553] error log buffer (63 bytes): error   : Binary format for key='0', ident='' is not recognize
    2023-12-08 15:24:09.063350: W external/xla/xla/service/gpu/runtime/support.cc:58] Intercepted XLA runtime error:
    INTERNAL: Failed to load PTX text as a module: CUDA_ERROR_INVALID_IMAGE: device kernel image is invalid
    2023-12-08 15:24:09.063374: E external/xla/xla/pjrt/pjrt_stream_executor_client.cc:2593] Execution of replica 0 failed: INTERNAL: Failed to execute XLA Runtime executable: run time error: custom call 'xla.gpu.func.launch' failed: Failed to load PTX text as a module: CUDA_ERROR_INVALID_IMAGE: device kernel image is invalid; current tracing scope: concatenate.1; current profiling annotation: XlaModule:#hlo_module=jit_matrix_power,program_id=2#.

    XlaRuntimeError: INTERNAL: Failed to execute XLA Runtime executable: run time error: custom call 'xla.gpu.func.launch' failed: Failed to load PTX text as a module: CUDA_ERROR_INVALID_IMAGE: device kernel image is invalid; current tracing scope: concatenate.1; current profiling annotation: XlaModule:#hlo_module=jit_matrix_power,program_id=2#.
