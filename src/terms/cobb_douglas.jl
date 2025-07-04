export CobbDouglas


"""
    CobbDouglas

Cobb-Douglas term for ALL variables in a vector `x`.

`g(x) = x[1]^α[1] * x[2]^α[2] * ... * x[n]^α[n]`

where ∑_{i=1}^n α[i] is NOT necessarily equal to 1.

- Input size : `x` ∈ R^n
- Output size: R^1

## Fields
- `D::Int`: the number of variables in the input vector `x`.
- `αs::Vector{Float64}`: the shares of each variable in the Cobb-Douglas term.

## Notes
- By default, the shares α[i] are set to 1/n, i.e. equal shares.
- This term is a family that has type parameter `D` which indicates the number 
of variables in the input vector `x`. A wrong `D` will result in numerical error
when incorporating the term into a GAM. The constructor requries `D` as a type
parameter, e.g. `CobbDouglas{3}(αs = [0.5, 0.3, 0.2])` for a 3-variable 
Cobb-Douglas term.
"""
struct CobbDouglas <: AbstractTerm
    D ::Int
    αs::Vector{Float64}
    function CobbDouglas(; D::Int = 2, αs::Vector{Float64} = fill(1/D,D))
        @assert D > 0 "D must be a positive integer"
        @assert length(αs) == D "αs must have length D"

        new(D,αs)
    end
end
# ------------------------------------------------------------------------------
function Base.length(g::CobbDouglas, n::Int):Int
    return 1
end
# ------------------------------------------------------------------------------
function (g::CobbDouglas)(x::AbstractVector)::Vector{Float64}
    return Float64[prod(x .^ g.αs),]
end
# ------------------------------------------------------------------------------
function ∂(g::CobbDouglas, x::AbstractVector)::Matrix{Float64}
    grad = zeros(Float64, 1, g.D)
    for i in 1:g.D
        grad[1, i] = g.αs[i] * prod(x .^ g.αs) / x[i]
    end
    return grad
end
# ------------------------------------------------------------------------------
function ∂2(
    g::CobbDouglas,
    x::AbstractVector
)::Vector{Matrix{Float64}}

    # NOTE: Cobb-Douglas is special, k_i = 1 always
    n = length(x)
    @assert n == g.D "x must have length $(g.D) to match CobbDouglas"

    fval = prod(x .^ g.αs) # pre-conditioning for speed
    nomi = g.αs .* g.αs' - diagm(g.αs)
    deno = x .* x'

    return Matrix{Float64}[
        fval .* nomi ./ deno
    ]
end
# ------------------------------------------------------------------------------
function todict(g::CobbDouglas)::Dict{String,Any}
    return Dict{String,Any}(
        "type" => "CobbDouglas",
        "D" => g.D,
        "alpha" => g.αs
    )
end
# ------------------------------------------------------------------------------
function fromdict_CobbDouglas(di::Dict{String,Any})::CobbDouglas
    return CobbDouglas(
        D  = di["D"],
        αs = di["alpha"] |> Vector{Float64}
    )
end