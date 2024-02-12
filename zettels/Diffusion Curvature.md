---
share_link: https://share.note.sx/ucz5vuki#c8UEiiuvurGCYqtRnLQpX8KwlkcSBz0fD0kySCbE5zM
share_updated: 2024-02-09T12:31:25-05:00
---
```tasks
not done
(tags includes #projects/DiffusionCurvature) OR (heading includes #projects/DiffusionCurvature)
```



[ClickUp](https://app.clickup.com/9013032587/v/li/901300236665)
# Flow Model for Neural Flattening

###### Current next steps: #projects/DiffusionCurvature 
- [x] Implement radial flattening autoencoder ✅ 2024-01-22
- [x] Incorporate that loss into MIOFlow ✅ 2024-01-25
- [ ] Test both together and independently on toy data of neuroflattening versus uniform sampling. How well does neural flattening actually match the distribution of data?
	- [x] Write benchmarking scripts ✅ 2024-01-25
	- [x] Benchmark the RFAE with just distance loss +/- others ✅ 2024-01-25
	- [ ] Benchmark MIOFlow here

## MioFlow for Neural Flattening
[ClickUp](https://app.clickup.com/t/86a1wx0c4)

[[2024-02-07#A Uniform Density Loss for MIOFlattener]]



# A Neural Flattening Agenda

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

# Experimenting
## Next Steps for Neural Flattening

Does the Radial Flattening Autoencoder play well with MIOFlow? Is MIOFlow a necessary addition to this framework? Does it even make sense to first perform the logarithmic map and then try to disperse the points? Wouldn’t this *destroy* whatever curvature information we have from radial flattening?
- Perhaps in a low data regime this can be justified. Mapping the points into a euclidean space will make the diffusion *somewhat* flatter, though curvature-derived sampling effects will still bias the diffusion to behave more positively/negatively than in a true flat space.

My frustration with this whole neural flattening thing is that *it has never much sense*. The problem isn’t well specified and is likely fundamentally impossible – and we’ve now, in this project, wasted months dithering about it. I shouldn’t be trying to perfect the neural flattener – just get it to a stage where it can be benchmarked.

### For this, we just need #projects/DiffusionCurvature 
- [x] Incorporate the radial flattening autoencoder into MIOFlow, to counter the most obvious failings of that model ✅ 2024-02-08 ^i6zjw9
- [x] Try to fix the RAE’s problem with breaking up negative curvature. ✅ 2024-02-08
	- Perhaps the kernel size just has to be larger.
- [ ] Run it again in the [[Sampling Semifinals]], both with and without the radial flattening ae
- [ ] Incorporate into the diffusion curvature core.

MIOFlattener with Radial AE really seems to be working.

![[Workshop/21-SUMRY-Curvature/diffusion-curvature/zettels/Library/download.gif]]


![[Workshop/21-SUMRY-Curvature/diffusion-curvature/zettels/Library/download 1.gif]]




## Curvature Verification

The most critical missing piece. *Is the method working – is it reliably detecting sign?* Presently, we have no clear answer. [[Diffusion Curvature and the Curvature Colosseum]] endeavored to answer this question, but I’m not confident it was sampling correctly in high dimensions – and neither our method, nor Abby’s, was picking up any signs. 

Repairing this, and creating an independent validation via some other dataset – these are my clearest priorities.

### Rebuild the Curvature Colosseum
- [ ] Rebuild the curvature colosseum with more restrictions on the polynomial coefficients
	- [ ] Perform sampling sanity checks; ensure at least $N$ points in any subspace.
	- [ ] Perhaps throw out generated surfaces with extreme curvatures.

## Klein Bottle Image Data

Used by [[@sritharan2021ComputingRiemannianCurvature]]. This is a dataset of known negative curvature.

## More Competitors for Benchmarking


## Denoising of Results