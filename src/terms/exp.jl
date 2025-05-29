"""
    Exponential(; degrees::Vector{Int} = [1,])

Exponential terms of each element of `x` vector.

- Input size : `x` ∈ R^n
- Output size: R^n

Allocation:

[exp(x[1]), exp(x[2]), ...]

Attention: the returned Jacobian and Hessian are DENSE. The sparse version may
come in the future.
"""
struct Exponential <: AbstractTerm
    function Exponential()
        new()
    end
end
# ------------------------------------------------------------------------------
function Base.length(g::Exponential, n::Int)::Int
    return n
end
# ------------------------------------------------------------------------------
function (g::Exponential)(x::AbstractVector)::Vector{Float64}
    return exp.(x)
end
# ------------------------------------------------------------------------------
function ∂(g::Exponential, x::AbstractVector)::Matrix{Float64}
    return diagm(exp.(x))
end
# ------------------------------------------------------------------------------
function ∂2(g::Exponential, x::AbstractVector)::Vector{Matrix{Float64}}
    n = length(x)
    res = Matrix{Float64}[zeros(Float64, n, n) for _ in 1:n]
    for i in 1:n
        res[i][i, i] = exp(x[i])
    end
    return res
end