# StyleGAN on Swift for TensorFlow

- [Paper of StyleGAN](https://arxiv.org/abs/1812.04948)
- [Official implementation](https://github.com/NVlabs/stylegan)

## Problem

The loss functions used in StyleGAN are wgan-gp loss and non-staurating loss with gradient penalty.
S4TF doesn't support higher order differentiation currently so we can't implement both of them.

There's two losses implemented (non-saturating loss and LSGAN loss). 
I tried but mode collapses with these losses.