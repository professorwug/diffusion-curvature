---
id: dea18259-7117-44c3-ba56-0f7f794cfd6c
share_link: https://share.note.sx/ucz5vuki#c8UEiiuvurGCYqtRnLQpX8KwlkcSBz0fD0kySCbE5zM
share_updated: 2024-02-09T12:31:25-05:00
---
# Definition

Given samples $X \subseteq M$ and a flattening map $\Phi: X \rightarrow \mathbb{R}^d$,

The $t$-step *Diffusion Curvature* of $x$ is
$$
k_t(x)=1-\frac{W_1\left(\delta_x, p_X^t(x)\right)}{W_1\left(\delta_x, p_{\Phi(x)}^t(x)\right)}
$$

Where $p_X^t$ is the t-step random walk over $X$, and $p_{\Phi(X)}^t$ is the same over the flattened points $\Phi(X)$. In both cases, the $W_1$ distance is taken with respect to manifold distances.





```tasks
not done
(tags includes #projects/DiffusionCurvature) OR (heading includes #projects/DiffusionCurvature)
```

| [ClickUp](https://app.clickup.com/9013032587/v/li/901300236665)  | [Diffusion Curvature](omnifocus:///folder/dVHuiaIwheP) | 



# Flow Model for Neural Flattening

###### Current next steps: #projects/DiffusionCurvature 
- [x] Implement radial flattening autoencoder ✅ 2024-01-22
- [x] Incorporate that loss into MIOFlow ✅ 2024-01-25
- [x] Test both together and independently on toy data of neuroflattening versus uniform sampling. How well does neural flattening actually match the distribution of data?
	- [x] Write benchmarking scripts ✅ 2024-01-25
	- [x] Benchmark the RFAE with just distance loss +/- others ✅ 2024-01-25
	- [x] Benchmark MIOFlow here ✅ 2024-02-19
- [x] Incorporate the winning neural flattener into Diffusion Curvature core ✅ 2024-02-20

## MioFlow for Neural Flattening
[ClickUp](https://app.clickup.com/t/86a1wx0c4)

[[2024-02-07#A Uniform Density Loss for MIOFlattener]]



## Radial Flattening Autoencoder

As discussed with lan, the ideal flattening of $N(x) \in M$ has

1. Radial distances $d_{M}(x, y)=d_{E}(\varphi(x), \varphi(y))$ $\forall y \in M$. This is the primary objective.
2. Neighborhoods of each $y$ should be roughly preserved, ie. $K L D\left(A_{E}, A_{\mu}\right)$ $<\varepsilon$.
3. The space should be flat, egg. MMD $(\varphi(x)$. $\left.\mathbb{R}^{\sigma}\right)<\varepsilon$.

Let's try making the geodesic auto encoder focused on 1 & 2 . Perhaps this will be sufficient; if not, we'll use MIOFlow to add 3 (while also keeping 1 as a penalty in MIOFlow).

#### Early Results
![[CleanShot 2024-01-22 at 16.56.39.png]]

Using just the radial distance loss, with a sufficiently low learning rate, we get a convincing flattening:
![[download 12.png]]
Using the other losses oddly doesn't seem to add much:
![[download 13.png]]


## Radial Flattening AE + MIOFlow does admirably

Does the Radial Flattening Autoencoder play well with MIOFlow? Is MIOFlow a necessary addition to this framework? Does it even make sense to first perform the logarithmic map and then try to disperse the points? Wouldn’t this *destroy* whatever curvature information we have from radial flattening?
- Perhaps in a low data regime this can be justified. Mapping the points into a euclidean space will make the diffusion *somewhat* flatter, though curvature-derived sampling effects will still bias the diffusion to behave more positively/negatively than in a true flat space.

My frustration with this whole neural flattening thing is that *it has never much sense*. The problem isn’t well specified and is likely fundamentally impossible – and we’ve now, in this project, wasted months dithering about it. I shouldn’t be trying to perfect the neural flattener – just get it to a stage where it can be benchmarked.

### Final steps #projects/DiffusionCurvature 
- [x] Incorporate the radial flattening autoencoder into MIOFlow, to counter the most obvious failings of that model ✅ 2024-02-08 ^i6zjw9
- [x] Try to fix the RAE’s problem with breaking up negative curvature. ✅ 2024-02-08
	- Perhaps the kernel size just has to be larger.
- [x] Run it again in the [[Sampling Semifinals]], both with and without the radial flattening ae ✅ 2024-02-19

MIOFlattener with Radial AE really seems to be working.

![[Workshop/21-SUMRY-Curvature/diffusion-curvature/zettels/Library/download.gif]]


![[Workshop/21-SUMRY-Curvature/diffusion-curvature/zettels/Library/download 1.gif]]





# Denoising of Results


# More Competitors for Benchmarking



# Curvature Verification

The most critical missing piece. *Is the method working – is it reliably detecting sign?* Presently, we have no clear answer. [[Diffusion Curvature and the Curvature Colosseum]] endeavored to answer this question, but I’m not confident it was sampling correctly in high dimensions – and neither our method, nor Abby’s, was picking up any signs. 

Repairing this, and creating an independent validation via some other dataset – these are my clearest priorities.

## Operation Sadsphere
![[2024-02-20#^f4pp7p]]

[Saddle Sphere Ablations](http://athomia:8888/notebooks/21-SUMRY-Curvature/diffusion-curvature/nbs/library/datasets/saddle-sphere-ablations.ipynb) implements, using a new 'self-evaluating dataset' class which I will surely use more.

Next steps here:
1. 


### Rebuild the Curvature Colosseum
- [ ] Rebuild the curvature colosseum with more restrictions on the polynomial coefficients
	- [ ] Perform sampling sanity checks; ensure at least $N$ points in any subspace.
	- [ ] Perhaps throw out generated surfaces with extreme curvatures.



## Klein Bottle Image Data

Used by [[@sritharan2021ComputingRiemannianCurvature]]. This is a dataset of known negative curvature.