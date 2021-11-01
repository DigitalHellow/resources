# Resources
A list of Style transfer resources.

## Basics
Style transfer algorithms strive to create a new image with the content of one or more images using the style of another.
It first gained popularity by the paper [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576.pdf) published by Gatys et al.,
although most implementations use the one proposed by Johnson et al. ([Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf)),
such as PyTorch's [Fast Neural Style](https://github.com/pytorch/examples/tree/master/fast_neural_style).

<p align="center">
  <img src="imgs/gatys.png" />
</p>

We can separate style transfer algorithms into two classes:
- Single image style transfer
- Video or real time style transfer

In this document we will outline methods for both classes.

## Single Image Style Transfer
This class of algorithms aims to pass the style of one image into another's content. 

- [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/pdf/1603.08155.pdf) ([Implementation](https://github.com/pytorch/examples/tree/master/fast_neural_style)) is a relatively fast algorithm, which is used as the basis by many others.
It uses a pre-trained convolutional neural network (CNN) to train a transformer network on a style image and learn to pass its style into 
a content image.

<p align="center">
  <img src="imgs/johnson.png" />
</p>

- [Feature Visualization](https://distill.pub/2017/feature-visualization/) is another method of style transfer, although based
on the same principles of Johnson et al. implementation. It is done by peeking at the network's layers to see which ones activates
for the base content and style image, and training a new CNN which merges the desired activation layers. Many methods
can be seen [at this Lucid article](https://distill.pub/2018/differentiable-parameterizations/)([Lucid](https://github.com/tensorflow/lucid) might be deprecated, 
as it seems to have been lacking support in the recent years). One thing to note is the ability of these methods to also train style transfer from a style image 
into a [3D model](https://distill.pub/2018/differentiable-parameterizations/#section-style-transfer-3d).

<p align="center">
  <img src="imgs/lucid_fft.png" width="500"/>
  <img src="imgs/lucid_pat.png" width="500"/>
  <img src="imgs/lucid_3d.png" width="500"/>
</p>

- [Style Transfer by Relaxed Optimal Transport and Self-Similarity - STROTSS](https://arxiv.org/pdf/1904.12785.pdf)([Implementation](https://github.com/pytorch/examples/tree/master/fast_neural_style)) uses a new approach where
a mask is also used as an input at training time. The regions denoted withing the mask will have the style of the style image, while the other regions
will be the same as the content image. It also allows for aesthetic guidance- that is, you can select patches of the mask to have the same style,
allowing for more user control.

<p align="center">
  <img src="imgs/strotss_1.png" />
  <img src="imgs/strotss_2.png" />
</p>

### Adjustable Style Transfer
Style transfer has many hyperparameters to tune during training time. A poor choice of values of these will result in a need to train the
entire network again. But worse still, we cannot fine-tune an already good model to make it better - we'd need to train it all again. That's
where adjustable style transfer comes into play. They aim to increase or decrease the style image weight at inference time, allowing to 
make small modifications of the generated image after training.

- [Real-Time Style Transfer With Strength Control](https://arxiv.org/pdf/1904.08643.pdf)([implementation](https://github.com/victorkitov/style-transfer-with-strength-control)) 
by Kitov, uses the transformer network of Johnson et al. (with a few modifications besides the ones described here) but adds a stylization strength parameter, 
&alpha;, as an input parameter to control the style transfer at inference time. It also adds another trainable parameter &beta; to the residual blocks of the network, 
where &alpha; is used.

<p align="center">
  <img src="imgs/strength.png" />
</p>

- [Adjustable Real-time Style Transfer](https://arxiv.org/pdf/1811.08560.pdf)([implementation](https://github.com/gnhdnb/adjustable-real-time-style-transfer))
by Babaeizadeh, Ghiasi, also uses the same algorithm of Johnson et al., but adds a new network, &Lambda;, whose inputs &alpha;, generates an output
(&beta; and &gamma) which is fed into the transformer network to control the style strength of the generated images. 

<p align="center">
  <img src="imgs/babaeizadeh.png" />
</p>
