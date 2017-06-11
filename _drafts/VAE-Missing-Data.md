---
layout: post
title:  "Inferring missing data with a Variational Autoencoder"
categories: VAE missing data Gibbs
---
Recently people in our group have been discussing methods for filling in missing data using a latent variable model with some possibly approximate posterior distribution over the latents. In this post I'll discuss a few simple approaches, but firstly, let me specify the problem the problem we're trying to address...

<img style="float: right; height: 114px; margin:10px 10px 10px 10px" src="/images/model.png">Suppose we have a latent variable model, with hidden variables $$y$$ and observed variables $$x$$. Assume that we've already learned a good setting of the parameters of this model, and then we are presented with one or more new _partial_ observations, observations which have some missing data.

For example, we might be modelling MNIST digits, in which case full sample observations look something like this:

![Digits](/images/mnist_digits_full.png)

and we're given digits with half of the pixels randomly removed, which look like this:

![Fuzzy digits](/images/mnist_digits_partial50.png)

or maybe we're given something a bit more drastic, like these, which have had the top three quarters removed:

![Quarter digits](/images/mnist_digits_quarter.png)

We want to use our model to do inference on the missing pixels. That is, we want to know about the distribution $$p(x_{\mathrm{missing}}\mid x_{\mathrm{observed}})$$ — the distribution of the missing pixels, conditioned on what we have observed.

The first method for solving this problem which I'll discuss is, I think, the simplest, and is the approach taken by \cite{rezende}. It is an approximate form of Gibbs sampling.

### Gibbs Sampling
Gibbs sampling is a very general technique for sampling from a distribution over a number of variables. Suppose we have a joint distribution

$$
p(x_1, x_2, \ldots, x_N)
$$

which we would like to sample from. Suppose that we don't know of a direct way to sample from this distribution, but that we do have access to the conditional distributions of each variable conditioned on all the others, that is $$ p(x_n\mid x_1, \ldots, x_{n-1}, x_{n+1}, \ldots, x_N) $$ for each $$ n = 1, \ldots, N $$. To do Gibbs sampling, fistly initialise all of the variables $$x_1, \ldots, x_N$$ to any random values. Then, iterate through the variables one at a time, sampling each variable $$x_n$$ from the conditional distribution of that variable conditioned on the values of all the others. That is, for each $$n$$, you resample $$x_n$$, according to

$$
p(x_n\mid x_1, \ldots, x_{n-1}, x_{n+1}, \ldots, x_N).
$$

Asymptotically, as the number of iterations grows, under some mild assumptions, the values of $$x_1, \ldots, x_N$$ will be distributed according to the joint distribution $$ p(x_1, \ldots, x_N) $$, which is what we wanted. For more details, including proof of convergence to the target density, see section 29.5 of \cite{mackay}.

### Gibbs sampling the missing data
In our latent variable model, we can use Gibbs sampling to sample a pair $$x_{\mathrm{missing}}, y$$ from $$p(x_{\mathrm{missing}}, y\mid x_{\mathrm{observed}})$$
