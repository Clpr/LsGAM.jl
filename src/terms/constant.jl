export Constant, NegConstant

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



"""
    NegConstant

Negative constant term, always equal to -1. Useful in some contexts.
"""
struct NegConstant <: AbstractTerm
end
# ------------------------------------------------------------------------------
function Base.length(g::NegConstant, n::Int)::Int
    return 1
end
# ------------------------------------------------------------------------------
function (g::NegConstant)(x::AbstractVector)::Vector{Float64}
    return [-1.0,]
end
# ------------------------------------------------------------------------------
function ∂(g::NegConstant, x::AbstractVector)::Matrix{Float64}
    return zeros(Float64,1,length(x))
end
# ------------------------------------------------------------------------------
function ∂2(g::NegConstant, x::AbstractVector)::Vector{Matrix{Float64}}
    return Matrix{Float64}[zeros(Float64, length(x), length(x)),]
end