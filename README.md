# zi2zi: Learning Easter Asian character style with GAN
<p align="center">
  <img src="assets/intro.gif" alt="animation", style="width: 350px;"/>
</p>

## Introduction
Learning eastern asian language typeface with GAN. zi2zi(字到字, meaning from character to character) is an application and extension of the recent popular [pix2pix](https://github.com/phillipi/pix2pix) model to Chinese characters.

## Network Structure
![alt network](assets/network.png)

The network structure is based off pix2pix with the addition of category embedding and two other losses(category loss/constant loss). Details could be found in [this]() blog post.
## Gallery
### Ground Truth Comparison

<div style="text-align:center">
<img src="assets/compare3.png" alt="compare" style="width: 500px;"/>
</div>

### Brush Writing Fonts
<div style="text-align:center">
<img src="assets/cj_mix.png" alt="compare" style="width: 500px;"/>
</div>

### Korean
<div style="text-align:center">
<img src="assets/kr_mix.png" alt="compare" style="width: 500px;"/>
</div>

### Interpolation
![alt transition](assets/transition.png)

### Animation
<p align="center">
  <img src="assets/poem.gif" alt="animation", style="width: 250px;"/>
  <img src="assets/ko_wiki.gif" alt="animation", , style="width: 250px;"/>
</p>

<p align="center">
  <img src="assets/reddit_bonus_humor_easter_egg.gif" alt="easter egg", style="width: 350px;"/>
</p>


## How to Use
### Requirement
* Python 2.7
* CUDA
* cudnn
* Tensorflow >= 1.0.1
* Pillow(PIL)
* numpy >= 1.12.1
* scipy >= 0.18.1
* imageio

### Preprocess
### Experiment Layout
### Train
### Infer and Interpolate

## Acknowledgements
Code derived and rehashed from:

* [pix2pix-tensorflow](https://github.com/yenchenlin/pix2pix-tensorflow) by [yenchenlin](https://github.com/yenchenlin)
* [Domain Transfer Network](https://github.com/yunjey/domain-transfer-network) by [yunjey](https://github.com/yunjey)
* [ac-gan](https://github.com/buriburisuri/ac-gan) by [buriburisuri](https://github.com/buriburisuri)
* [dc-gan](https://github.com/carpedm20/DCGAN-tensorflow) by [carpedm20](https://github.com/carpedm20)
* [origianl pix2pix torch code](https://github.com/phillipi/pix2pix) by [phillipi](https://github.com/phillipi)

## License
Apache 2.0