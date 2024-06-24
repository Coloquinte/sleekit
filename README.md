# Bag of Tricks for NN Quantization

Neural network quantization is the process that compresses the weights in a neural network to use a smaller number representation.
This makes its representation smaller, both on disk and in memory, and can make the computation less expensive for accelerators, typically by using small integer weights for the coefficients.
At the same time, it reduces the precision of the computations, so that good algorithm design is necessary to maintain good quality.

This repository contains tools to research post-training neural networks quantization, with methods to improve over the current state-of-the-art.
It is purely for analysis purpose: complete implementations will be made available on other repositories.
Our main contributions are two simple improvements that are compatible with most quantization methods: an improved scaling method, and making better use of the bias during quantization.

## Quantization method

Sleekit uses a very generic quantization method. The steps to quantize a layer are:
* gathering sample data: we run the network on some data samples to gather statistical informations for each layer;
* chosing of a codebook: a codebook gives a limited number of values that can be represented, and we round the weights to one of the values in the codebook;
* scaling the weights: we apply a scaling factor so that the weights are close to the chosen codebook;
* optimizing the weights: to maintain a good quality for the neural network, we use a specialized algorithm to tweak the weights after rounding.

## Improvements

We present deveral generic improvements that can be applied to any quantization method.
They will target both the scaling step, to select better scaling factors, and the weight optimization step to reduce the layer error.

### Methodology: layer-per-layer analysis

To develop our methods we analyze the effect of quantization decisions on a per-layer basis.
Despite many previous works using network-level metrics, post-training quantization methods minimize the error at the layer level.
Analyzing the error at the layer level is therefore the natural approach.
Moreover, network-level metrics have a tendency to be noisy, can hide small quantization errors or on the contrary be over-sensitive to some layers.

Our baseline for comparison is the [GPTQ](https://arxiv.org/abs/2210.17323) algorithm with 3-bit and 1.5-bit weights.
We use GPTQ's given parameters for the heuristic (diagonal ordering and 1% dampening).
For the layer weights and metrics, we use layer statistics from a full accuracy run on several smaller networks (OPT-125M, OPT-350M, BLOOM-560M).
We compare the error introduced by the quantization with and without our methods.

### Trick 1: better scaling

A good scaling factor minimizes the error introduced by quantization.
The typical method is to chose a scaling factor that minimizes the mean squared error on the weights (MSE).
We introduce a more precise approach, that optimizes the layer's result directly.

For weight optimization, we already have access to an accurate measure of the layer's error (the hessian matrix $H$ obtained from input samples).
Our idea is to reuse it for scaling optimization.
We test three different approaches to scaling, and compare the layer error after applying GPTQ:
* minimizing the mean squared error after rounding to the nearest;
* using the full hessian matrix to compute the error, which is computationally expensive;
* using the diagonal of the hessian matrix to compute the error, which has the same computational cost as the MSE;
* using the full weight optimization to compute the error for each scaling value, which is extremely expensive but is theoretically optimal.

<div align="center">
<img src="results/scaling_1.5b.png" width=45%><img src="results/scaling_3b.png" width=45%>
</div>

The usual approach of minimizing the MSE yields results that are far from optimal.
Using the full hessian matrix or its diagonal yields similar results that are on average much better than MSE alone.
Results are far from the theoretical optimum, and even slightly degraded for some layers, leaving room for improvement. 


### Trick 2: combining with bias correction

[Bias correction](https://arxiv.org/abs/1810.05723) is a method used to reduce the impact of quantization on a layer.
Newer quantization methods behave much better, and it is not used much anymore.
However, it is compatible and there is no reason not to use both.
The effect of bias correction can even be integrated in the cost function used for weight optimization, using $H=\frac{1}{n} X^\intercal X -M^\intercal M$, where $X$ are the input samples and $M = \frac{1}{n}1^\intercal X$ is the average value of the samples for each input.

We test three different ways to update the bias:
* applying weight optimization alone (GPTQ) without bias correction;
* applying bias correction after weight optimization, yielding a slightly smaller layer error;
* taking the effect of bias correction into account during weight optimization.

<div align="center">
<img src="results/correction_1.5b.png" width=45%><img src="results/correction_3b.png" width=45%>
</div>

Adding back bias correction greatly improves certain layers, in particular some attention layers in all networks.
Unsurprisingly, it has more impact with a more agressive quantization and yields better result if taken into account for weight optimization.

### Trick 3: adding local search

The weight optimization problem is NP-hard, and can only be solved at scale in an approximate manner.
GPTQ provides a good heuristic for it, however the heuristic of choice to obtain good solutions to similar problems (QUBO) is a simple local search.
For this reason, we test the effect of applying a few local search moves after GPTQ, in a best-first manner.

<div align="center">
<img src="results/local_search_1.5b.png" width=45%><img src="results/local_search_3b.png" width=45%>
</div>

The effect of just a few local search moves is notable on many layers, and applying them after GPTQ can drastically reduce layer error.

### Minor tricks

Other tricks yield smaller but useful improvements:
* Using a different ordering for GPTQ. GPTQ makes rounding decisions for the weights in a greedy manner; they obtain a good ordering using the diagonal of the matrix in decreasing order. 
Instead, we multiply this value by the sum of squares of the quantization error; this takes better account of the effect of saturation.
* Using a different dampening for GPTQ. GPTQ does not behave well on ill-conditioned matrix, and adding a larger penalty term to the matrix paradoxically yields better results.
The original paper uses a 1% penalty, but penalties of 3-10% behave better, while removing the penalty altogether degrades results significantly.

<div align="center">
<img src="results/ordering_3b.png" width=40%><img src="results/dampening_3b.png" width=40%>
</div>

### The many tricks that do not work

The following approaches did not yield promising results and were abandoned:
* Improved codebooks: the data is far from being gaussian-distributed, but training a codebook naively on the data is not better than a [NF4 codebook](https://arxiv.org/abs/2305.14314).
A good codebook training needs to take the hessian (or its diagonal) into account.
* Entropy coding: it is tempting to combine codebook optimization with entropy coding to reduce storage needs. However, the gain in entropy is not huge compared to an error-optimized codebook, and does not seem worth the effort.
* GPTQ reordering: clever heuristic orderings for GPTQ based on the hessian matrix do not bring a reduction in layer error, compared to using its diagonal as the original paper does. We tested several variations using the diagonal of the inverse and pivoted Cholesky decompositions.
* More complex algorithms for weight optimization: it just doesn't scale, but if you want to go in this direction you probably want to use the [MQLib](https://github.com/MQLib/MQLib) as a solver.

### Putting it all together

Finally, we put these algorithms together in Sleekit:

1. the hessian matrix is modified to represent the effect of bias correction;
2. scaling is performed based on the hessian diagonal;
3. weight optimization uses our slightly improved ordering and dampening.

The computational cost of the algorithm is not increased so far compared to GPTQ: this is the "Sleekit light" version.

At the cost of additional computations we add the following for the "Sleekit heavy" version:

4. scaling is performed based on a weight optimization computation;
5. local search is performed during the final weight optimization for 1000 moves.

<div align="center">
<img src="results/compare_1.5b.png" width=45%><img src="results/compare_3b.png" width=45%>
</div>

Most of the improvement is due to the better scaling method, but the various methods stack well. Together, they yield to a reduced error on almost all layers.
On the other hand, a minority of layers experiences a huge improvement, with error reduced by 80% or more. It is still unclear what the impact is for the neural network as a whole.

## References

The algorithms in this repository build on the following works:
* [Bias correction](https://arxiv.org/abs/1810.05723) and [GPTQ](https://arxiv.org/abs/2210.17323) for the approach to weight quantization, as well as similar works such as [AdaRound](https://arxiv.org/abs/2004.10568), [AdaQuant](https://arxiv.org/abs/2006.10518), [OBQ](https://arxiv.org/abs/2208.11580) or [GPTVQ](https://arxiv.org/abs/2402.15319);
* [Lloyd](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) and [LBG](https://en.wikipedia.org/wiki/Linde%E2%80%93Buzo%E2%80%93Gray_algorithm) for the choice of quantization grids;
* The [GPTQ repository](https://github.com/IST-DASLab/gptq) was used for data and testing.
