export Interaction

"""
    Interaction()

Linear interaction term across multiple dimensional variables of a vector `x`.
There is no quadratic term e.g. `x[i]^2`.

Order/Allocation:
[
    x[1] * x[2],
    x[1] * x[3],
    ...
    x[1] * x[n],
    x[2] * x[3],
    x[2] * x[4],
    ...
    x[n-1] * x[n]
]

- Input size : `x` ∈ R^n
- Output size: R^{n * (n - 1) / 2}
"""
struct Interaction <: AbstractTerm
    function Interaction()
        new()
    end
end
# ------------------------------------------------------------------------------
function Base.length(g::Interaction, n::Int)::Int
    return n * (n - 1) ÷ 2
end
# ------------------------------------------------------------------------------
function (g::Interaction)(x::AbstractVector)::Vector{Float64}
    return Combinatorics.combinations(x, 2) .|> prod
end
# ------------------------------------------------------------------------------
function ∂(g::Interaction, x::AbstractVector)::Matrix{Float64}
    n = length(x)
    k = length(g, n)
    res = zeros(Float64, k, n)
    
    ctr = n - 1 # length counter
    rptr = 1    # row pointer
    cptr = 1    # col pointer
    while ctr > 0
        res[rptr:rptr+ctr-1, cptr] .= x[cptr+1:end]
        res[rptr:rptr+ctr-1, cptr+1:end] .= diagm(fill(x[cptr], ctr))
        rptr += ctr
        cptr += 1
        ctr -= 1
    end

    return res
end
# ------------------------------------------------------------------------------
function ∂2(g::Interaction, x::AbstractVector)::Vector{Matrix{Float64}}
    n = length(x)
    k = length(g, n)
    res = [zeros(Float64, n, n) for _ in 1:k]
    
    for (h, ij) in Combinatorics.combinations(1:n, 2) |> enumerate
        #= 
        this is Hessian for ∂²(x[i] * x[j]) / ∂ x[p] ∂ x[q],
        where p, q = 1,...,n

        the 2nd order derivative is non-zero only for:
        - p == i && q == j --> 1.0
        - p == j && q == i --> 1.0

        the rest is zero, so we can skip it
        =#
        res[h][ij[1], ij[2]] = 1.0
        res[h][ij[2], ij[1]] = 1.0
    end
    return res
end
# ------------------------------------------------------------------------------
function todict(g::Interaction)::Dict{String,Any}
    return Dict{String,Any}(
        "type" => "Interaction"
    )
end
# ------------------------------------------------------------------------------
function fromdict_Interaction(di::Dict{String,Any})::Interaction
    return Interaction()
end