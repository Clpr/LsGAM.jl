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






# ==============================================================================
"""
    chebT(x::Real, n::Int)::Float64

Compute the Chebyshev polynomial of the first kind of degree `n` at point `x`.
Using the Clenshaw recurrence relation.
"""
function chebT(x::Real, n::Int)::Float64
    @assert n >= 0 "Degree n must be non-negative"
    t0 = 1.0
    n == 0 && return t0
    t1 = x
    n == 1 && return t1
    tn = NaN
    for _ in 2:n
        tn = 2 * x * t1 - t0
        t0 = t1
        t1 = tn
    end
    return tn
end
# ------------------------------------------------------------------------------
"""
    chebU(x::Real, n::Int)::Float64

Compute the Chebyshev polynomial of the second kind of degree `n` at point `x`.
Using the Clenshaw recurrence relation.
"""
function chebU(x::Real, n::Int)::Float64
    @assert n >= 0 "Degree n must be non-negative"
    u0 = 1.0
    n == 0 && return u0
    u1 = 2 * x
    n == 1 && return u1
    un = NaN
    for _ in 2:n
        un = 2 * x * u1 - u0
        u0 = u1
        u1 = un
    end
    return un
end




