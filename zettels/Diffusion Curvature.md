[ClickUp](https://app.clickup.com/9013032587/v/li/901300236665)
# Flow Model for Neural Flattening

Current next steps:
- [x] Implement radial flattening autoencoder ✅ 2024-01-22
- [x] Incorporate that loss into MIOFlow ✅ 2024-01-25
- [ ] Test both together and independently on toy data of neuroflattening versus uniform sampling. How well does neural flattening actually match the distribution of data?
	- [x] Write benchmarking scripts ✅ 2024-01-25
	- [x] Benchmark the RFAE with just distance loss +/- others ✅ 2024-01-25
	- [ ] Benchmark MIOFlow her

## MioFlow for Neural Flattening
[ClickUp](https://app.clickup.com/t/86a1wx0c4)

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

# Denoising of Sampling-Induced Variations
