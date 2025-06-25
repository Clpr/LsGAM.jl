export FracionalPoly

"""
    FracionalPoly(; degrees::Vector{Float64} = [0.5,])

Polynomial terms of each element of `x` vector but with fractional degrees.

- Input size : `x` ∈ R^n
- Output size: R^{length(degrees) * n}

Allocation:

[x[1]^degrees[1], x[2]^degrees[1], ..., x[2]^degrees[1], x[2]^degrees[2], ...]

Attention: the returned Jacobian and Hessian are DENSE. The sparse version may
come in the future.

## Notes
- zero degree is not allowed, i.e. any 0 in `degrees=` will throw an error.
"""
struct FracionalPoly <: AbstractTerm
    degrees::Vector{Float64}
    function FracionalPoly(; degrees::Vector{Float64} = [0.5,])
        @assert all(degrees != 0) "Zero degree is not allowed in FracionalPoly."
        new(degrees)
    end
end
# ------------------------------------------------------------------------------
function Base.length(g::FracionalPoly, n::Int)::Int
    return length(g.degrees) * n
end
# ------------------------------------------------------------------------------
function (g::FracionalPoly)(x::AbstractVector)::Vector{Float64}
    res = Float64[]
    for dgr in g.degrees
        append!(res, x .^ dgr)
    end
    return res
end
# ------------------------------------------------------------------------------
function ∂(g::FracionalPoly, x::AbstractVector)::Matrix{Float64}
    return reduce(vcat, [
        diagm(dgr .* x .^ (dgr - 1))
        for dgr in g.degrees
    ])
end
# ------------------------------------------------------------------------------
function ∂2(g::FracionalPoly, x::AbstractVector)::Vector{Matrix{Float64}}
    n   = length(x)
    res = Matrix{Float64}[]
    for dgr in g.degrees, j in 1:n
        tmp = zeros(Float64, n, n)
        tmp[j,j] = dgr * (dgr - 1) * x[j] ^ (dgr - 2)
        push!(res, tmp)
    end
    return res
end