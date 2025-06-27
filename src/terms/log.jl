export Logarithm

"""
    Logarithm(; ϵ::Float64 = 1E-8)

`g(x) = [ln(x[1] + ϵ), ln(x[2] + ϵ), ...]`
"""
struct Logarithm <: AbstractTerm
    ϵ::Float64
    function Logarithm(; ϵ::Float64 = 0.0)
        new(ϵ)
    end
end
# ------------------------------------------------------------------------------
function Base.length(g::Logarithm, n::Int):Int
    return n
end
# ------------------------------------------------------------------------------
function (g::Logarithm)(x::AbstractVector)::Vector{Float64}
    return log.(x .+ g.ϵ)
end
# ------------------------------------------------------------------------------
function ∂(g::Logarithm, x::AbstractVector)::Matrix{Float64}
    return diagm(1.0 ./ (x .+ g.ϵ))
end
# ------------------------------------------------------------------------------
function ∂2(g::Logarithm, x::AbstractVector)::Vector{Matrix{Float64}}
    n = length(x)
    res = Matrix{Float64}[]
    for i in 1:n
        tmp = zeros(Float64, n, n)
        tmp[i,i] = - 1.0 / (x[i] + g.ϵ)^2
        push!(res,tmp)
    end
    return res
end
# ------------------------------------------------------------------------------
function todict(g::Logarithm)::Dict{String,Any}
    return Dict{String,Any}(
        "type" => "Logarithm",
        "ϵ" => g.ϵ
    )
end
# ------------------------------------------------------------------------------
function fromdict_Logarithm(di::Dict{String,Any})::Logarithm
    return Logarithm(ϵ = di["ϵ"] |> Float64)
end