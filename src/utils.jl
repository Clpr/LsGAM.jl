# UTILITIES
export R²


# ==============================================================================
function unitvec(n::Int, j::Int ; v::Float64 = 1.0)::Vector{Float64}
    x = zeros(Float64, n)
    x[j] = v
    return x
end
function spunitvec(n::Int, j::Int ; v::Float64 = 1.0)::SparseVector{Float64}
    return sparsevec([j,], [v,], n)
end
# ------------------------------------------------------------------------------
function R²(ytrue::AbstractVector, ypred::AbstractVector)::Float64
    n = length(ytrue)
    @assert n == length(ypred) "ytrue and ypred must have the same length"
    @assert n > 1 "R² requires at least two data points"
    ytrue_mean = sum(ytrue) / n

    ss_total    = sum((ytrue .- ytrue_mean).^2)
    ss_residual = sum((ytrue .- ypred).^2)

    if ss_total == 0.0
        # R² is undefined if all ytrue values are the same
        return Inf
    end

    return 1.0 - (ss_residual / ss_total)
end

