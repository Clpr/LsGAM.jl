"""
    Poly(; degrees::Vector{Int} = [1,])

Polynomial terms of each element of `x` vector.

- Input size : `x` ∈ R^n
- Output size: R^{length(degrees) * n}

Allocation:

[x[1]^degrees[1], x[2]^degrees[1], ..., x[2]^degrees[1], x[2]^degrees[2], ...]

Attention: the returned Jacobian and Hessian are DENSE. The sparse version may
come in the future.
"""
struct Poly <: AbstractTerm
    degrees::Vector{Int}
    function Poly(; degrees::Vector{Int} = [1,])
        new(degrees)
    end
end
# ------------------------------------------------------------------------------
function Base.length(g::Poly, n::Int)::Int
    return length(g.degrees) * n
end
# ------------------------------------------------------------------------------
function (g::Poly)(x::AbstractVector)::Vector{Float64}
    res = Float64[]
    for dgr in g.degrees
        append!(res, x .^ dgr)
    end
    return res
end
# ------------------------------------------------------------------------------
function ∂(g::Poly, x::AbstractVector)::Matrix{Float64}
    return reduce(vcat, [
        diagm(dgr .* x .^ (dgr - 1))
        for dgr in g.degrees
    ])
end
# ------------------------------------------------------------------------------
function ∂2(g::Poly, x::AbstractVector)::Vector{Matrix{Float64}}
    n   = length(x)
    res = Matrix{Float64}[]
    for dgr in g.degrees, j in 1:n
        tmp = zeros(Float64, n, n)
        tmp[j,j] = dgr * (dgr - 1) * x[j] ^ (dgr - 2)
        push!(res, tmp)
    end
    return res
end