---
title: 'MatDL: A Lightweight Deep Learning Library in MATLAB'
tags:
  - Machine Learning
  - Deep Learning
  - Neural Networks
authors:
 - name: Haytham M. Fayek
   orcid: 0000-0002-1840-7605
   affiliation: 1
affiliations:
 - name: RMIT University
   index: 1
date: 16 September 2017
bibliography: paper.bib
---

# Summary

*MatDL* [@Fayek2017] is an open-source lightweight deep learning [@LeCun2015; @Goodfellow2016] library native in MATLAB that implements some most commonly used deep learning algorithms. 
The library comprises functions that implement the following: (1) basic building blocks of modern neural networks such as affine transformations, convolutions, nonlinear operations, dropout, batch normalization, etc.; (2) popular architectures such as deep neural networks (DNNs), convolutional neural networks (ConvNets), and recurrent neural networks (RNNs) and their variant, the long short-term memory (LSTM) RNNs; (3) optimizers such stochastic gradient descent (SGD), RMSProp and ADAM; as well as (4) boilerplate functions for training, gradients checking, etc.
Most of these functions can run on a CPU or a MATLAB-compatible CUDA-enabled GPU.
It is straight forward to use the low-level functions to experiment with or test new architectures or training algorithms, or alternatively use the provided models for applied deep learning research.

*MatDL* was inspired by Stanford's CS231n [@CS231n] and Torch [@Collobert2011], and is conceptually similar to Keras [@Chollet2015] and Lasagne [@Lasagne2015], but unlike these libraries, it is natively implemented in MATLAB.
This makes it convenient in cases where MATLAB is preferred, or if it is required to be closely linked with other libraries written in MATLAB or Octave.
*MatDL* is ideal for rapid machine learning research and experimentation, specially with small datasets, as it was designed with an emphasis on modularity, flexibility and extensibility.

*MatDL* is MIT-licensed and can be retrieved from [GitHub](https://github.com/haythamfayek/MatDL) ([https://github.com/haythamfayek/MatDL](https://github.com/haythamfayek/MatDL)).

# References
