# Improved neural network quantization

Neural network quantization is the process that compresses the weights in a neural network to use a smaller number representation.
This makes its representation smaller, both on disk and in memory, and can make the computation less expensive for accelerators, typically by using small integer weights for the coefficients.
At the same time, it reduces the precision of the computations, so that good algorithm design is necessary to maintain good quality.

This repository contains tools for post-training neural networks quantization. It puts together several state-of-the-art algorithms, and analyzes potential improvements.

## Quantization method

Sleekit uses a generic quantization method. The steps to quantize a layer are:
* gathering sample data: we run the network on some data samples to gather statistical informations for each layer;
* chosing of a codebook: a codebook gives a limited number of values that can be represented, and we round the weights to one of the values in the codebook;
* scaling the weights: we apply a scaling factor so that the weights are close to the chosen codebook;
* optimizing the weights: to maintain a good quality for the neural network, we use a specialized algorithm to tweak the weights after rounding.

Our baseline is the [GPTQ](https://arxiv.org/abs/2210.17323) algorithm, which provides a very efficient weight optimization.
We then introduce improvements to these steps that result in much better behaviour.

## Improvements

To benchmark improvements over existing methods, we analyze the effect of quantization decisions on a per-layer basis.
Since post-training quantization operates at the layer level, this gives us a much more precise view of quantization behaviour than network-level metrics.
We export layer statistics from a full accuracy run on several smaller networks (OPT-125M, OPT-350M, BLOOM-560M).

### Scaling

A good scaling factor minimizes the error introduced by quantization.
This is usually done quickly but roughly, and the current methods chose a scaling factor that minimizes the mean squared error on the weights (MSE).
We introduce a more precise approach, that optimizes the layer's result directly.

For weight optimization, we have access to an accurate measure of the layer's error (the hessian), that we integrate directly with the scaling optimization.
We test three different approaches to scaling, and compare the layer error after applying GPTQ:
* minimizing the mean squared error;
* minimizing the exact error using the full hessian, which is computationally more expensive;
* minimizing an approximate error using the hessian's diagonal, which has the same computational cost as the MSE.

<!-- TODO: figure and conclusion -->

### Bias correction

Bias correction is [a method](https://arxiv.org/abs/1810.05723) used to reduce the impact of quantization on a layer.
It is not used in association with newer quantization methods, which provide better benefits.

We test three different ways to update the bias:
* applying weight optimization alone (GPTQ) without bias correction;
* applying bias correction after weight optimization, yielding a slightly smaller layer error;
* taking the effect of bias correction into account during weight optimization with a modified hessian.

<!-- TODO: figure and conclusion -->

### Things that do not work

The following approaches did not yield promising results and were abandoned:
* Improved codebooks: the data is far from being gaussian-distributed, but training a codebook naively is not better than a NF4 codebook.
* Entropy coding: it is tempting to combine codebook optimization with entropy coding to reduce storage needs. However, the gain in entropy is not huge compared to an error-optimized codebook, and does not seem worth the effort.
* GPTQ reordering: other heuristic orderings for GPTQ do not bring a consistent reduction in layer error compared to using the diagonal of the hessian as the original paper does.
* More complex algorithms for weight optimization: it just doesn't scale, but if you want to go in this direction you probably want to use the [MQLib](https://github.com/MQLib/MQLib) as a solver.

## References

The algorithms in this repository build on the following works:
* [Bias correction](https://arxiv.org/abs/1810.05723) and [GPTQ](https://arxiv.org/abs/2210.17323) for the approach to weight quantization, as well as similar works such as [AdaRound](https://arxiv.org/abs/2004.10568), [AdaQuant](https://arxiv.org/abs/2006.10518), [OBQ](https://arxiv.org/abs/2208.11580) or [GPTVQ](https://arxiv.org/abs/2402.15319);
* [Lloyd](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) and [LBG](https://en.wikipedia.org/wiki/Linde%E2%80%93Buzo%E2%80%93Gray_algorithm) for the choice of quantization grids;
* The [GPTQ repository](https://github.com/IST-DASLab/gptq) was used for data and testing.
