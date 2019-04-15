# Implementation of Relational Deep Reinforcement Learning

This Repository is implementation of [Relational Deep Reinforcement Learning](https://arxiv.org/pdf/1806.01830.pdf) to Breakout Environment.

The Reinforcement Learning Algorithm is [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)

## Configuration

* This paper requires heavy computation power.
* Left Figure is the map of attention which is produced by self-attention.
* Though the paper developed 100 environments for experiment, the implementer of this repository created only 16 environments with the limitation of computer resources. So sometimes it's exactly the performance and sometimes it's not.
* If you want to see more significant attention map, just control CNN function to have less strides and more filters. In this repository, 84, 84 images are processed to have 19, 19 because of my computation limit.

## Initial Training status

<div align="center">
  <img src="source/before_train.gif" width="50%" height='300'>
</div>

## During Training

<div align="center">
  <img src="source/after_train.gif" width="50%" height='300'>
</div>

## Tensorboard

<div align="center">
  <img src="source/out.png" width="50%" height='300'>
</div>
