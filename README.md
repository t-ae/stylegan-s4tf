# StyleGAN on Swift for TensorFlow

StyleGAN implementation on Swift for TensorFlow, up to 256x256 image genetarion.

- [Paper of StyleGAN](https://arxiv.org/abs/1812.04948)
- [Official implementation](https://github.com/NVlabs/stylegan)

## Unimplemented features

- Style mixing
- Truncation trick

## Loss function

Today's Swift for TensorFlow doesn't support higher-order differentiation, whereas the loss functions described in StyleGAN paper require it.

Instead of them, I employed LSGAN loss described in [ProGAN paper](https://arxiv.org/abs/1710.10196).

## Result

After 700000 steps of training on [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) (took about 100 hours with single RTX2080).
![face](https://user-images.githubusercontent.com/12446914/64338274-5548b200-d01c-11e9-8ac1-0e1bd05904df.png)
