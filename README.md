# LsGAM.jl
Implementation of least-sqaure generalized additive model for multi-dimensional smooth function approximation.

```julia
pkg> add "https://github.com/Clpr/LsGAM.jl.git"
```

## GAM with vector-valued terms

### Formula

Consider a scalar function $f(x):\mathbb{R}^n \to \mathbb{R}$ where $x$ is an $n$-vector.
We consider the following vector-valued modified GAM approximation:

$$
\hat{f}(x) := \sum_{i=1}^m g_i(x)^T \beta_i
$$

where $g_i(x):\mathbb{R}^{n} \to \mathbb{R}^{k_i}$ is a vector function ("term") that takes $x$ and returns a $k_i$-vector; the $\beta_i \in\mathbb{R}^{k_i}$ is the approximation coefficient of the $i$-th term.
Such a function approximation can be represented by a tuple $(n, \{g_i(x)\}_{i=1}^m, \{\beta_i\}_{i=1}^m)$.



With data $(\mathbf{X} \in \mathbb{R}^{N\times n}, \mathbf{y} \in \mathbb{R}^{N})$, one can fit the approximation with least square method. If we stack the returned values by each $g_i(x)$ into a vector $z$

$$
z := \begin{bmatrix}
g_1(x) \\
\vdots \\
g_m(x) \\
\end{bmatrix} \in \mathbb{R}^{\sum_{i=1}^m k_i}
$$

Then the problem becomes a typical linear least squares fitting which can be efficiently solved using mature numerical solvers. In this case, $R^2$ is good to measure the fitting goodness.

After fitting $\hat{f}(x)$, depending on the smoothness of the terms, one can define Jacobian $J_f(x) \in \mathbb{R}^{n}$ and Hessian $H_f(x) \in\mathbb{R}^{n \times n}$ respectively.
The chain rule applies to each term due to the additivity:

$$
J_f(x) = \sum_{i=1}^m (\nabla_x g_i(x))^T \cdot \beta_i
$$

where $\nabla_x g_i(x) \in \mathbb{R}^{k_i \times n}$. 


> **Remark**
> The Hessian of a vector-valued function $g(x) : \mathbb{R}^{n} \to \mathbb{R}^{k_i}$ is a sequence/list/vector of $n\times n$ matrices in which each matrix corresponds to the Hessian of the $j$-th element of $g(x)$ wrt vector $x$. Let's denote the $j$-th element of $g(x)$ with $g^j(x) \in \mathbb{R}$, and the $j$-th element of $\beta$ be $\beta^j \in\mathbb{R}$.

The Hessian is then:

$$
H_f(x) = \sum_{i=1}^m \sum_{j=1}^{k_i}   \nabla^2_x g^j_i(x)  \cdot \beta^j_i
$$


### Analyticity 

The core idea of GAM is to approximate high-dimensional complex object $f(x)$.
In the context of optimization, such $f(x)$ is usually the objective function whose Jacobian and Hessian are important in the iterations. However, finite difference is slow and inaccurate, while the auto differentiation may not work due to some data structure conflictions.
The core idea of this package is to conform a GAM with pre-designed smooth terms.
These terms are simple and have analytical Jacobian and Hessian.
Then, the linearity in the formula allows us to efficiently compose the final Jacobian and Hessian using the chain rule above.

The currently available term functions:

- `Constant()`: constant term
- `Poly(degrees = [1,2,3,...])`: dimension-wise ordinary polynomial terms
- `Logarithm(ϵ = 1E-8)`: natural log transformation with a small shift to avoid absolute zero
- `Interaction()`: all distinct pairwise interaction terms between the dimension elements of $x$

These terms are good enough for many typical applications e.g. approximating a high-dimensional value function in economic research. More terms are on the way.


### Clamping

Some times, the output of $\hat{f}(x)$ is supposed be within a specific range $[y_\text{min},y_\text{max}]$. This package allows clamping/bounding of the output values, which acutally adds an extra "layer" on the top of the GAM evaluation. However, this may break the smoothness in the regions where the clamping happens. The package automatically switches to numerical differentiation in this case.



## Usage


TBD






















