export Constant

"""
    Constant

Constant term, always equal to 1.
"""
struct Constant <: AbstractTerm
end
# ------------------------------------------------------------------------------
function Base.length(g::Constant, n::Int)::Int
    return 1
end
# ------------------------------------------------------------------------------
function (g::Constant)(x::AbstractVector)::Vector{Float64}
    return [1.0,]
end
# ------------------------------------------------------------------------------
function ∂(g::Constant, x::AbstractVector)::Matrix{Float64}
    return zeros(Float64,1,length(x))
end
# ------------------------------------------------------------------------------
function ∂2(g::Constant, x::AbstractVector)::Vector{Matrix{Float64}}
    return Matrix{Float64}[zeros(Float64, length(x), length(x)),]
end