export CRRA


"""
    CRRA()

CRRA (Constant Relative Risk Aversion) term for each variable in a vector `x`.

`g(x) = [x[1]^(1-γ)/(1-γ), x[2]^(1-γ)/(1-γ), ...]`


- Input size : `x` ∈ R^n
- Output size: R^n

## Notes
- γ != 1 required; otherwise, use `Logarithm()` instead.
- In convex GAM, to denote x^(-m) where m > 0 polynomials (round or fractional),
Use CRRA() where γ > 1.
"""
struct CRRA <: AbstractTerm
    γ::Float64
    function CRRA(; γ::Float64 = 2.0)
        @assert γ != 1.0 "Use Logarithm() instead for γ = 1.0"
        new(γ)
    end
end
# ------------------------------------------------------------------------------
function Base.length(g::CRRA, n::Int):Int
    return n
end
# ------------------------------------------------------------------------------
function (g::CRRA)(x::AbstractVector)::Vector{Float64}
    return x .^ (1.0 .- g.γ) ./ (1.0 .- g.γ)
end
# ------------------------------------------------------------------------------
function ∂(g::CRRA, x::AbstractVector)::Matrix{Float64}
    return diagm(x .^ (-g.γ))
end
# ------------------------------------------------------------------------------
function ∂2(g::CRRA, x::AbstractVector)::Vector{Matrix{Float64}}
    n = length(x)
    res = Matrix{Float64}[]
    for i in 1:n
        tmp = zeros(Float64, n, n)
        tmp[i,i] = - g.γ * x[i] .^ (-g.γ - 1.0)
        push!(res,tmp)
    end
    return res
end
# ------------------------------------------------------------------------------
function todict(g::CRRA)::Dict{String,Any}
    return Dict{String,Any}(
        "type" => "CRRA",
        "gamma" => g.γ
    )
end
# ------------------------------------------------------------------------------
function fromdict_CRRA(di::Dict{String,Any})::CRRA
    return CRRA(γ = di["gamma"])
end