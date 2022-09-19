# Parametric Integer Programming

This is a minimum working example
of [Parametric Integer Programming](https://en.wikipedia.org/wiki/Parametric_programming),
a means to solving [Integer Linear Programs](https://en.wikipedia.org/wiki/Integer_programming) (ILPs) with parameters
$\theta$ in either the objective or the right-hand sides of the constraints (through duality).

$$ \begin{aligned}
J^{*}(\theta )=&\min _{x\in \mathbb {R} ^{n}}f(x,\theta )\\
&{\text{subject to }}g(x,\theta )\leq 0.\\
&\theta \in \Theta \subset \mathbb {R} ^{m}
\end{aligned} $$

The principal idea is to run a simplex solver but *symbolically*, i.e.,
leave the parameters unevaluated until the pivot step, at which point you branch depending on the sign of the
coefficient (and already explored branches).
The solution is then a piecewise affine function of the parameters.

[For example](https://github.com/makslevental/pip/blob/da0eff59b532fc2d1d0094b45f262f45b6cb4732/tests.py#L154), the
system

$$ \begin{aligned}
& \text{minimize}_{\lambda_1, \lambda_2} && z = x_1\lambda_1 + x_2\lambda_2 \\
& \text{subject to} && \lambda_1 + \lambda_2 \leq 5 \\
& && -\lambda_1 \leq 1 \\
& && -\lambda_2 \leq 2 \\
& && -\lambda_1 + \lambda_2 \leq 0 \\
& && \lambda_1, \lambda_2 \succcurlyeq 0 \\
& \text{and} && \boldsymbol {\lambda}, \mathbf {x} \in \mathbb {Z} ^{n},\end{aligned} $$

induces the following tree structure

<p align="center">
  <img width="500" src="docs/tree.png" alt="">
</p>

and produces the following solution

<p align="center">
  <img width="500" src="docs/soln.png" alt="">
</p>

# Bibliography

1. [Parametric integer programming 1988 by Paul Feautrier](http://www.numdam.org/item/RO_1988__22_3_243_0.pdf)
2. [Section 5.1 of New Algorithmics for Polyhedral Calculus via Parametric Linear Programming by Alexandre Maréchal](https://hal.archives-ouvertes.fr/tel-01695086v3/document) (this is the best gentle introduction to the ideas)
2. [FPL: A Fast Presburger Library](https://grosser.science/FPL)

# Disclaimer

This repo is purely for illustration and experimentation. The code is guaranteed to be neither correct nor fast nor original
i.e., it has, in fact, been cobbled together from various sources.
Maybe one day it'll be something...
