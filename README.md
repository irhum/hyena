# Hyena
This repository provides a JAX/Flax implementation of the Hyena architecture introduced in [Poli et. al. (2023)](https://arxiv.org/abs/2302.10866). A full training run of a small 1.5M parameter model, on the Shakespeare dataset can be found in the included `intro.ipynb`. This achieves a best validation loss of ~1.45, on par with the results in [nanoGPT](https://github.com/karpathy/nanoGPT).

## Details
Specifically, the following is implemented:

* The Hyena layer itself can be found in `hyena/hyena.py` as `HyenaOperator`
    * The efficient, FFT-based convolution is implemented in the `fftconv` method, providing an O(N log N) complexity in sequence length. This is used for training, and for the pre-fill stage during inference.
        * Caching is also implemented, which means this is called only once during inference pre-fill, with the subsequent individual tokens being computed using the alternate implementation (see below).
    * An alternate implementation, having O(N) complexity *per token* is provided for the auto-regressive decoding stage during inference. This is implemented in the `inference_conv` method. It will be particularly faster when generating a small number of tokens from a very large input (e.g. a full document).
* A standard Decoder tower is implemented in `hyena/decoder.py` as `Decoder`. The implementation is largely similar to the one in [nanoGPT](https://github.com/karpathy/nanoGPT), with the self-attention layers swapped out with Hyena layers.