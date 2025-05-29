"""
    LsGAM.jl

This package implements a special class of Generalized Additive Models (GAMs)
that are based on least squares fitting. It is designed to be efficient and
flexible, allowing for the inclusion of various types of terms, including
vector-valued terms (e.g. interaction terms) and scalar terms.

## Math

Consider a function `f(x): R^n -> R` that we want to approximate using a GAM. 
The model can be expressed as:

`f(x) ≈ β₀ + ∑_{i=1}^m g_i(x_i)`

where `g_i(x):R^n -> R^{k_i}` are the individual terms, and `β₀` is a scalar 
bias term. The terms `g_i` can be scalar or vector-valued, allowing for
flexible modeling of the relationship between the input `x` and the output 
`f(x)`.

"""
module LsGAM
# ==============================================================================
using LinearAlgebra
using SparseArrays
import Printf: @printf, @sprintf


using Combinatorics            # for interaction terms
import MultivariateStats as mvs # for least squares fitting


# ==============================================================================
abstract type AbstractGAM{N} <: Any end
abstract type AbstractTerm   <: Any end

include("utils.jl")
include("interface.jl")


# Terms ========================================================================

# constant/intercept/bias term
include("terms/constant.jl")

# polynomial term(s) (integer order)
include("terms/poly.jl")

# logarithm term(s)
include("terms/log.jl")

# 1st order interaction term(s)
include("terms/cross.jl")

# exponential term(s)
include("terms/exp.jl")



# the follows coming soon ....

# sigmoid/logit term(s)










# GAM formula ==================================================================

# standard GAM model with optional clamping
include("formula/standard.jl")



















# ==============================================================================
end # module LsGAM