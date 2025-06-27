export ConvexGAM


"""
    ConvexGAM{N} <: AbstractGAM{N}

A convex Generalized Additive Model (GAM) with non-negative coefficients for all
terms except the constant term. This model is suitable for cases where the users
would like to ensure the concavity/convexity of the model, such as in economics.

## Notes
- Be careful about the choice of `Constant` and `NegConstant` terms, as they can
largely affect the model's behavior. Try both and choose the one that fits your 
data best.


## Example
```julia
lgam = include("src/LsGAM.jl")

# simulated data
X = rand(100, 2)
y = -1.0 ./ (X[:,1] .+ X[:,2]) .+ randn(100) .* 0.1


# define: convex model
eq = lgam.ConvexGAM(
    X, y,
    [
        lgam.Constant(),
        lgam.Poly(degrees = [-1,]),
        lgam.Logarithm(),
        lgam.InvExponential(),
    ],
    ylim = (-Inf,0.0),
)
println(eq)
println(eq.β)

# test: negative constant term if knowing the data property
eq_inv = lgam.ConvexGAM(
    X, y,
    [
        lgam.NegConstant(),
        lgam.Poly(degrees = [-1,]),
        lgam.Logarithm(),
        lgam.InvExponential(),
    ],
    ylim = (-Inf,0.0),
)
println(eq_inv)
println(eq_inv.β)


# visualization
import Plots as plt

fig1 = plt.surface(
    LinRange(0.001,1,50),
    LinRange(0.001,1,50),
    (x,y) -> eq([x,y]),
    camera   = (-30,30),
    alpha    = 0.5,
    colorbar = false,
    title    = "ConvexGAM, Constant()",
); 
plt.scatter!(
    fig1,
    X[:,1], X[:,2], y,
    markersize = 5,
    markercolor = :red,
    label = "data",
)
fig2 = plt.surface(
    LinRange(0.001,1,50),
    LinRange(0.001,1,50),
    (x,y) -> eq_inv([x,y]),
    camera   = (-30,30),
    alpha    = 0.5,
    colorbar = false,
    title    = "ConvexGAM, NegConstant()",
);
plt.scatter!(
    fig2,
    X[:,1], X[:,2], y,
    markersize = 5,
    markercolor = :red,
    label = "data",
)
plt.plot(fig1,fig2,layout = (1,2), size = (800,400))

```
"""
mutable struct ConvexGAM{N} <: AbstractGAM{N}

    terms::Vector{AbstractTerm}    # list of terms g_i(x_i)
    β    ::Vector{Vector{Float64}} # vector of coefficients β_i for each term
    m    ::Int                # number of terms g_i
    r2   ::Float64            # pseudo R² of the model, i.e. the goodness of fit
    ylim ::NTuple{2, Float64} # limits for the predicted values

    function ConvexGAM{N}(
        terms::Vector{AbstractTerm} ;
        ylim ::NTuple{2, Float64} = (-Inf, Inf),
    ) where N

        m  = length(terms)
        β  = Vector{Float64}[zeros(Float64, length(term,N)) for term in terms]
        r2 = NaN
        return new{N}(terms, β, m, r2, ylim)
    end
end
# ------------------------------------------------------------------------------
function Base.show(io::IO, eq::ConvexGAM{N}) where N
    npars = [length(coef) for coef in eq.β] |> sum
    @printf(
        io, 
        "ConvexGAM{R^%d --> R} with %d terms, total %d parameters\n", 
        N, eq.m, npars
    )
    @printf(io, "  ylim = [%.2f, %.2f]\n", eq.ylim...)
    @printf(io, "  R2   = %.4f\n", eq.r2)
    println(io, "  Formula: f(x) = ")
    for (i, term) in enumerate(eq.terms)
        @printf(io, "  + %s\n", typeof(term))
    end
    return nothing
end
# ------------------------------------------------------------------------------
function isclamped(f::ConvexGAM{N}) where N
    return isfinite(f.ylim[1]) || isfinite(f.ylim[2])
end
# ------------------------------------------------------------------------------
function fit!(
    f::ConvexGAM{N}, 
    X::AbstractMatrix, 
    y::AbstractVector ;

    algorithm::Symbol = :fnnls, # :nnls, :fnnls, :pivot
) where N
    
    # TODO: use Optim.jl or other optimization library that support sign
    #       restrictions on the coefficients.
    # NOTE: recommend to use `NonNegLeastSquares.jl` for non-negative LS fitting

    # expand: design matrix
    Greg = stack(f, X)

    # fit: non-negative least squares
    βvec = nnls.nonneg_lsq(Greg, y, alg = algorithm) |> vec

    # split: stacked coefficients to individual terms
    update!(f, βvec)

    # compute: pseudo R²
    ypred = Float64[f(x) for x in eachrow(X)]
    f.r2  = R²(y, ypred)

    return nothing
end
# ------------------------------------------------------------------------------
function ConvexGAM(
    X     ::AbstractMatrix,
    y     ::AbstractVector,
    terms ::Vector{AbstractTerm} ;

    ylim     ::NTuple{2, Float64} = (-Inf, Inf),
    algorithm::Symbol = :fnnls, # :nnls, :fnnls, :pivot
)

    nobs, N = size(X)
    @assert length(y) == nobs "X and y must have the same number of obs/rows"
    @assert N > 0 "X must have at least one column"

    f = ConvexGAM{N}(terms, ylim = ylim)
    fit!(f, X, y, algorithm = algorithm)

    return f
end
# ------------------------------------------------------------------------------
"""
    save_ConvexGAM!(f::StandardGAM{N}, fpath::String)::Nothing where N

Save the GAM model `f` to a JSON file at `fpath`. The path should be valid and
refer to a file rather than a directory. Overwrite the file if it exists. For
aesthetic reason, the exported JSON file is human readable and pretty-printed.

## Example
```julia
X = rand(5000,2); Y = sum(X,dims = 2) |> vec
f = gam.ConvexGAM(
    X, Y,
    [
        gam.Constant(),
        gam.Poly(degrees = [1,2]),
        gam.Logarithm(ϵ = 1E-8),
        gam.Exponential(),
        gam.CobbDouglas(D = 2),
    ]
)
gam.save_ConvexGAM!(f, "trash.json")
```
"""
function save_ConvexGAM!(f::ConvexGAM{N}, fpath::String)::Nothing where N
    di = Dict{String, Any}(
        "type"   => "ConvexGAM",
        "type_N" => N, 

        "m"    => f.m,
        "r2"   => f.r2,

        # specially handle `ylim` as it may contain `-Inf` or `Inf`
        # tuples will be vectors in JSON
        "ylim" => string.(f.ylim),

        "terms" => Dict{String,Any}[todict(term) for term in f.terms],
        "β"     => Vector{Float64}[coef for coef in f.β],
    )

    open(fpath, "w") do fio
        JSON3.pretty(fio, di)
    end
end
# ------------------------------------------------------------------------------
"""
    load_ConvexGAM(fpath::String)::ConvexGAM

Load a ConvexGAM model from a JSON file at `fpath`.

## Example
```julia
# run the example in `save_ConvexGAM!` first
f2 = gam.load_ConvexGAM("trash.json")
```
"""
function load_ConvexGAM(fpath::String)::ConvexGAM
    di = JSON3.read(fpath, Dict{String, Any})

    # assume: the type is always "ConvexGAM"
    N    = Int(di["type_N"])
    r2   = Float64(di["r2"])
    m    = Int(di["m"])
    ylim = (
        parse(Float64, di["ylim"][1]),
        parse(Float64, di["ylim"][2]),
    )

    # load terms & coefficients
    terms = AbstractTerm[
        fromdict(di_term)
        for di_term in di["terms"]
    ]

    f = ConvexGAM{N}(terms, ylim = ylim)
    for i in 1:m
        f.β[i] .= di["β"][i]
    end

    f.m    = m
    f.r2   = r2
    f.ylim = ylim

    return f
end



