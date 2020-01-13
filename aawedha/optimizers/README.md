# diffGrad-tf

This is a Tensorflow version of the diffGrad optimizer from the paper [diffGrad: An Optimization Method for 
Convolutional Neural Networks](https://arxiv.org/abs/1909.11015).

This is version one from the paper. If you'd like version two, simply remove `math_ops.abs` from line 85 (making it
`1.0 / (1.0 + math_ops.exp(-(prev_g - grad)))`). I haven't implemented versions 3-5 but please feel free to contribute.
