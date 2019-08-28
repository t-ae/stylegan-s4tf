# StyleGAN on Swift for TensorFlow

- [Paper of StyleGAN](https://arxiv.org/abs/1812.04948)
- [Official implementation](https://github.com/NVlabs/stylegan)

## Unimplemented features

- Style mixing
- Truncation trick

## Loss function

Both loss functions used in StyleGAN paper require gradient penalty, which is not available for today's Swift for TensorFlow.

Instead of them, I employed LSGAN loss described in ProGAN paper.