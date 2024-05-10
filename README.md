# Fast and accurate neural network quantization

Neural network quantization is the process that compresses the weights in a neural network to use a smaller number representation.
This makes its representation smaller, both on disk and in memory, and can make the computation less expensive for accelerators, typically by using small integer weights for the coefficients.
At the same time, it reduces the precision of the computations, so that good algorithm design is necessary to maintain good quality.

This repository puts together several state-of-the-art algorithms to perform neural networks quantization post-training.
On top of it, we focus on the compression of neural networks weights for storage and online transmission.
This is particularly important in bandwidth-limited settings, like edge computing.

## Quantization method

Sleekit uses a generic quantization method. There are several steps to it:
* gathering sample data: we run the network on some data samples to gather statistical informations for each layer;
* chosing of a codebook: a codebook gives a limited number of values that can be represented, and we round the weights to one of the values in the codebook;
* scaling the weights: we apply a scaling factor so that the weights are close to the chosen codebook;
* optimizing the weights: to maintain a good quality for the neural network, we use a specialized algorithm to tweak the weights after rounding.

### Choice of codebook

Common quantization methods will use small integer weights for the codebook, but a good codebook is taylored to the data.
Since we are aiming for the smallest size, we provide codebooks optimized for entropy coding. 

### Weight scaling

A good scaling factor minimizes the error introduced by quantization.
This is usually done quickly but roughly: the error is evaluated on the weights.
With sleekit, we introduce a more precise approach: the error is evaluated on the layer's result.

### Weight optimization

The data received by a neural network layer contains a lot of correlation.
In practice, this means that we can compensate part of the rounding errors by taking advantage of this correlation.
Sleekit improves a little on the existing fast algorithms, and introduces a way to combine two of them for better results. 

## References

The algorithms in this repository build on the following works:
* [Bias correction](https://arxiv.org/abs/1810.05723), [AdaRound](https://arxiv.org/abs/2004.10568), [AdaQuant](https://arxiv.org/abs/2006.10518), [OBQ](https://arxiv.org/abs/2208.11580), [GPTQ](https://arxiv.org/abs/2210.17323) for the approach to weight quantization;
* Codebook design ([Lloyd](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) and [LBG](https://en.wikipedia.org/wiki/Linde%E2%80%93Buzo%E2%80%93Gray_algorithm)) and [Entropy-constrained Vector Quantization](https://www.researchgate.net/publication/236340572_Entropy-Constrained_Vector_Quantization) for the choice of quantization grids;
* [zstd](https://github.com/facebook/zstd) and [FSE](https://github.com/Cyan4973/FiniteStateEntropy) for the final compression stage.
