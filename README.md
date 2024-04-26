# Fast and accurate neural network quantization

This repository provides algorithms to quantize neural networks efficiently.
The focus is on distribution and storage of the neural networks, more than fast inference.

## References

The algorithms in this repository build on the following quantization algorithms:
* [AdaRound](https://arxiv.org/abs/2004.10568), [AdaQuant](https://arxiv.org/abs/2006.10518), [OBQ](https://arxiv.org/abs/2208.11580), [GPTQ](https://arxiv.org/abs/2210.17323) for the approach to weight quantization;
* Codebook design ([Lloyd](https://en.wikipedia.org/wiki/Lloyd%27s_algorithm) and [LBG](https://en.wikipedia.org/wiki/Linde%E2%80%93Buzo%E2%80%93Gray_algorithm)) and [Entropy-constrained Vector Quantization](https://www.researchgate.net/publication/236340572_Entropy-Constrained_Vector_Quantization) for the choice of quantization grids;
* [zstd](https://github.com/facebook/zstd) and [FSE](https://github.com/Cyan4973/FiniteStateEntropy) for the final compression stage.
