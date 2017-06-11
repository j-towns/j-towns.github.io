---
layout: post
title:  "A new trick for calculating Jacobian vector products"
categories: Autodiff optimization deep learning autograd theano
---

Last week I was involved in a [heated discussion thread](https://github.com/HIPS/autograd/pull/175) over on the [Autograd](https://github.com/HIPS/autograd) issue tracker. I'd recently been working on an implementation of forward mode automatic differentiation, which fits into Autograd's system for differentiating Python/Numpy code. Our discussion was about the usefulness of forward mode, which is equivalent to [Theano's Rop](http://deeplearning.net/software/theano/tutorial/gradients.html#r-operator) and in the general case is used to calculate directional derivatives, or equivalently for calculating _Jacobian vector products_.

Reverse mode automatic differentiation (also known as 'backpropagation'), is well known to be extremely useful for calculating gradients of complicated functions. In the general case, reverse mode can be used to calculate the Jacobian of a function _left multiplied_ by a vector. In the thread linked above, and in the Autograd codebase, any function which does this reverse mode vector-Jacobian product computation is abbreviated to _vjp_. The usefulness of a vjp operator, which takes a function as input and returns a vjp function (the familiar grad operator $$\nabla$$ is a special case of such an operator), is clearly beyond dispute. Such operators have now been implemented in different forms in many many software packages: TensorFlow, Theano, MXNet, Chainer, Torch, PyTorch, Caffe... etc etc.

But how useful are the _Jacobian-vector products_ (or jvps, as opposed to vjps), which are calculated by _forward mode_ (as opposed to reverse mode), particularly in machine learning? Well they may be useful as a necessary step for efficiently calculating Hessian-vector products (hvps), which in turn are used for second order optimization (see e.g. [this paper](http://www.cs.toronto.edu/~jmartens/docs/Deep_HessianFree.pdf)), although as I was arguing in the thread linked above, in an idealised implementation you can obtain an equivalent hvp computation by composing two reverse mode vjp operators (Lops in Theano).

Later in the thread we were discussing another very specific use case for forward mode, that of computing generalised Gauss Newton matrix-vector products, when we happened upon a _new trick_: a method for calculating jvps by composing two reverse mode vjps! This could render specialised code for forward mode jvps redundant. The trick is simple. I'll demonstrate it first mathematically and then with Theano code.

# The trick
Firstly, note that our reverse mode operator takes a function as input and returns a function for computing vector-Jacobian products. In mathematical notation, if we have a function

$$f:\mathbb{R}^m \rightarrow \mathbb{R}^n$$

then

$$\mathrm{vjp}(f)(v, x) = v^\top \frac{\partial f}{\partial x}(x)$$

where $$\frac{\partial f}{\partial x}$$ is shorthand for the _Jacobian matrix_ of $$f$$:

$$\frac{\partial f}{\partial x} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1}&\cdots&\frac{\partial f_1}{\partial x_m}\\
\vdots&\ddots&\vdots\\
\frac{\partial f_n}{\partial x_1}&\cdots&\frac{\partial f_n}{\partial x_m}
                                    \end{bmatrix}.$$

Now if we treat $$x$$ as a constant, and consider the _transpose_ of the above, which I will denote $$g$$, we have

$$
g(v) = \left[\mathrm{vjp}(f)(v, x)\right]^\top = \left[\frac{\partial f}{\partial x}(x)\right]^\top v.
$$

Note that $$g$$ is a _linear_ function of $$v$$. The Jacobian of $$g$$ is therefore constant (w.r.t. $$v$$), and equal to $$\left[\frac{\partial f}{\partial x}(x)\right]^\top$$, the transpose of the Jacobian of $$f$$.

So what happens when we apply the vjp operator to $$g$$? We get

$$
\begin{align}
\mathrm{vjp}(g)(u, v) &= u^\top \frac{\partial g}{\partial v}(v) \\
&=u^\top\left[\frac{\partial f}{\partial x}(x)\right]^\top \\
&=\left[ \left[ \frac{\partial f}{\partial x}(x)\right] u \right]^\top
\end{align}
$$

The Jacobian of $$f$$ is being right-multiplied by the vector $$u$$ inside the bracket, and taking the transpose of the whole of the above yields

$$
\left[\mathrm{vjp}(g)(u, v)\right]^\top = \left[ \frac{\partial f}{\partial x}(x)\right] u,
$$

the Jacobian of $$f$$ right multiplied by the vector $$u$$ — a Jacobian-vector product.

# Example implementation with Theano
In Theano the jvp operator is `theano.tensor.Rop` and the vjp operator is `theano.tensor.Lop`. I'm going to implement an 'alternative Rop' using only `theano.tensor.Lop`, and demonstrate that the computation graphs that it produces are the same as those produced by `theano.tensor.Rop`.

Firstly import `theano.tensor`:
```python
In [1]: import theano.tensor as T
```

and let's take a quick look at the API for `Rop`, which we'll want to copy for our alternative implementation:

```python
In [2]: T.Rop?
Signature: T.Rop(f, wrt, eval_points)
Docstring:
Computes the R operation on `f` wrt to `wrt` evaluated at points given
in `eval_points`. Mathematically this stands for the jacobian of `f` wrt
to `wrt` right muliplied by the eval points.
```
The important thing here is: `T.Rop(f, wrt, eval_points)` stands for 'the jacobian of `f` wrt to `wrt` right muliplied by the eval points'. So `wrt` was what I denoted $$x$$ above, and `eval_points` was denoted $$u$$.

I prefer my notation, so I'm going to use `x` and `u` for those variables. The signature of our alternative_Rop is going to look like this:

```python
def alternative_Rop(f, x, u):
```

The `theano.tensor.Lop` API matches that of `Rop`. So `T.Lop(f, wrt, eval_points)` evaluates the Jacobian product of `f` with respect to `wrt`, left multiplied by the vector `eval_points`. Carefully tracing through the equations above, we can implement our alternative Rop. It's pretty simple:
```python
In [3]: def alternative_Rop(f, x, u):
   ...:     v = f.type('v')       # Dummy variable v of same type as f
   ...:     g = T.Lop(f, x, v)    # Jacobian of f left multiplied by v
   ...:     return T.Lop(g, v, u)
   ...:
```
Note that we don't need any transposes because the output of Theano's Lop actually comes ready transposed.

Let's test this out on a real function and check that it gives us the same result as Theano's default Rop. Firstly define some input variables and a function:
```python
In [4]: x = T.vector('x')
   ...: W = T.matrix('W')
   ...: b = T.vector('b')
   ...: f = T.tanh(T.dot(W, x) + b)
```
and a variable to dot with the Jacobian of `f`
```python
In [5]: u = T.vector('u')
```
Then apply the original Rop and our alternative:
```python
In [6]: jvp = T.Rop(f, x, u)
   ...: alternative_jvp = alternative_Rop(f, x, u)
```
and compile them into Python functions
```python
In [7]: import theano
   ...: jvp = theano.function([x, W, b, u], jvp)
   ...: alternative_jvp = theano.function([x, W, b, u], alternative_jvp)
```
Then evaluate the two functions to check they output the same thing:
```python
In [8]: import numpy as np

In [9]: x_val = np.random.randn(3)
   ...: W_val = np.random.randn(3, 3)
   ...: b_val = np.random.randn(3)
   ...: u_val = np.random.randn(3)

In [10]: jvp(x_val, W_val, b_val, u_val)
Out[10]: array([-0.95290885,  0.71727834, -0.38178233])

In [11]: alternative_jvp(x_val, W_val, b_val, u_val)
Out[11]: array([-0.95290885,  0.71727834, -0.38178233])
```
The different versions seem to be doing the same computation, and we can verify this by rendering their computation graphs.

Firstly the __default Rop__
```python
In [12]: theano.printing.pydotprint(jvp, outfile='./jvp_theano_Rop.png', var_with_name_simple=True)
```
![Theano Rop graph](/images/jvp_theano_Rop.png)
and the __alternative Rop__
```python
In [13]: theano.printing.pydotprint(alternative_jvp, outfile='./jvp_alternative_Rop.png', var_with_name_simple=True)
```
![Alternative Rop graph](/images/jvp_alternative_Rop.png)
Perhaps it's not blindingly obvious, but after a careful look at those graphs it's clear that the same computation is being carried out, and crucially the dummy variable `v` is nowhere to be seen in the graph of the alternative implementation. This is because Theano knows that `g` is linear as a function of `v` and severs the connections when the second `Lop` derivative is applied.

# Conclusions and notes
