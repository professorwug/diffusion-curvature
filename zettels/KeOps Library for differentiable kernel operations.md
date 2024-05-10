# KeOps Library for differentiable kernel operations

[getkeops/keops: KErnel OPerationS, on CPUs and GPUs, with autodiff and without memory overflows](https://github.com/getkeops/keops)

Currently, [[Drafting/21-SUMRY-Curvature/Diffusion Curvature/Diffusion Curvature]] uses an unholy combination of dense and sparse matrices. Unholy because, while the graph constructions *should* be sparse, my differentiable matmul engine (jax) doesn’t handle sparse matrices. This problem is deeper than I’d realized:

> (the sparse matrix encoding) **does not stream well on GPUs**: parallel computing devices are wired to perform _block-wise_ memory accesses and have a hard time dealing with lists of _random_ indices (in,jn). As a consequence, when compared with dense arrays, sparse encodings only speed up computations for matrices that have **less than 1% non-zero coefficients**.

And while that may sometimes be true for graphs, it rapidly becomes false under many steps of diffusion.

Hence [[Drafting/21-SUMRY-Curvature/Diffusion Curvature/Diffusion Curvature]] is in a pickle. The adjacency matrices it needs to work with are often so large that they can only be used when sparse – but it also needs to *power* these matrices, which, to use hardware acceleration, requires them to be dense.

Fortunately, there appears to be a solution:

> *Symbolic matrices*. KeOps provides another solution to speed up tensor programs. Our key remark is that most of the large arrays that are used in machine learning and applied mathematics share a common mathematical structure. ==Distance matrices, kernel matrices, point cloud convolutions and attention layers can all be described as symbolic tensors==: given two collections of vectors (xi) and (yj), their coefficients Mi,j at location (i,j) are given by mathematical formulas F(xi,yj) that are evaluated on data samples xi and yj. 

> These objects are not "sparse" in the traditional sense… but can nevertheless be described efficiently using a mathematical formula F and relatively small data arrays (xi) and (yj). The main purpose of the KeOps library is to provide support for this abstraction with all the perks of a deep learning library:
> - A transparent interface with CPU and GPU integration.
> - Numerous tutorials and benchmarks.
> - Full support for automatic differentiation, batch processing and approximate computations.



