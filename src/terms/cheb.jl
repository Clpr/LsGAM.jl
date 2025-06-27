export ChebyshevT



"""
    ChebyshevT(; degree::Int = 2)

Chebyshev polynomial of the first kind of degree `n`. `x` is supposed to be in
the [-1,1] range. Do normalization before training if necessary.

- Input size: `x` ∈ R^n
- Output size: R^n

Allocation:

[chebT(x[1], degree), chebT(x[2], degree), ...]

## Attention:
- Full collinearity if `degree == 0` as it creates `n` constant terms.
- Full collinearity with `Poly(degree = [1,])` when the `degree` is 1. Exclude
the `Poly` term if you want to use `ChebyshevT` for polynomial regression.
- Use list comprehension to construct a sequence of ChebyshevT terms if doing
standard polynomial regression, e.g. `[ChebyshevT(degree = d) for d in 1:10]`.
"""
struct ChebyshevT <: AbstractTerm
    degree::Int
    function ChebyshevT(; degree::Int = 2)
        @assert degree > 0 "Degree degree must be > 0 due to the collinearity."
        new(degree)
    end
end
# ------------------------------------------------------------------------------
function Base.length(g::ChebyshevT, n::Int)::Int
    return n
end
# ------------------------------------------------------------------------------
function (g::ChebyshevT)(x::AbstractVector)::Vector{Float64}
    return chebT.(x, g.degree)
end
# ------------------------------------------------------------------------------
function ∂(g::ChebyshevT, x::AbstractVector)::Matrix{Float64}
    n = length(x)
    if g.degree == 1
        return diagm(ones(n))
    else
        return diagm(g.degree .* chebU.(x, g.degree - 1))
    end
end
# ------------------------------------------------------------------------------
function ∂2(g::ChebyshevT, x::AbstractVector)::Vector{Matrix{Float64}}
    n = length(x)
    res = Matrix{Float64}[zeros(Float64, n, n) for _ in 1:n]
    if g.degree == 1
        nothing # need to do as the Hessian is initialized to zero
    else
        for i in 1:n
            # if x[i] is non-distinguishable from 1 or -1 (the distance is lower
            # than the machine epsilon `eps()`, then `x^2-1` is numerically
            # identical to zero which leads to NaN. So we use the LHopital's
            # rule to compute the limit.
            _lim = (g.degree^4 - g.degree^2) / 3.0
            res[i][i, i] = if abs(x[i] - 1.0) < eps()
                _lim
            elseif abs(x[i] + 1.0) < eps()
                (-1)^g.degree * _lim
            else
                g.degree * (
                    (g.degree+1.0) * chebT(x[i],g.degree) - chebU(x[i],g.degree)
                ) / (x[i]^2 - 1.0)             
            end
        end
    end
    return res
end
# ------------------------------------------------------------------------------
function todict(g::ChebyshevT)::Dict{String,Any}
    return Dict{String,Any}(
        "type" => "ChebyshevT",
        "degree" => g.degree
    )
end
# ------------------------------------------------------------------------------
function fromdict_ChebyshevT(di::Dict{String,Any})::ChebyshevT
    @assert haskey(di, "degree") "Missing 'degree' key in dictionary."
    return ChebyshevT(degree = di["degree"] |> Int)
end






