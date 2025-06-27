export StandardGAM

"""
    StandardGAM{N}

Standard Generalized Additive Model (GAM) with `N`-vector input.


## Example
```julia
lgam = include("src/LsGAM.jl")

# simulated data
X = rand(100, 2)
y = sin.(X[:,1]) + cos.(X[:,2]) + randn(100) * 0.1

# define: blank model
eq = lgam.StandardGAM{2}(
    [
        lgam.Constant(),
        lgam.Poly(degrees = [1,2]),
        lgam.Logarithm(),
        lgam.Exponential(),
    ],
    ylim = (-Inf,Inf),
)

# fit: model to data
lgam.fit!(eq, X, y)
println(eq)

# define: directly fit
eq = lgam.StandardGAM(
    X, y,
    [
        lgam.Constant(),
        lgam.Poly(degrees = [1,2]),
        lgam.Logarithm(),
        lgam.Exponential(),
    ],
    ylim = (-Inf,Inf),
)
println(eq)

# test: other methods
let x0 = rand(2)
    eq(x0)

    # gradient: analytical vs. numerical
    [lgam.∂(eq, x0) lgam.jacobian(eq, x0)] |> display
    
    # Hessian: analytical vs. numerical
    [lgam.∂2(eq, x0), lgam.hessian(eq, x0)] |> display

    # expand: features
    stack(eq, rand(2))
    stack(eq, rand(100,2))

    # extract: regression coefficients of the design matrix
    coefficients = lgam.coefficients(eq)

    # update: regression coefficients
    lgam.update!(eq, rand(sum([
        length(term,ndims(eq)) for term in eq.terms
    ])))

    # check if the output is clamped 
    eq.ylim = (-Inf,Inf)
    @assert !lgam.isclamped(eq) "Opps!"
    eq.ylim = (-0.5,0.5)
    @assert lgam.isclamped(eq) "Opps too!"

end
```
"""
mutable struct StandardGAM{N} <: AbstractGAM{N}

    terms::Vector{AbstractTerm}    # list of terms g_i(x_i)
    β    ::Vector{Vector{Float64}} # vector of coefficients β_i for each term
    m    ::Int                # number of terms g_i
    r2   ::Float64            # pseudo R² of the model, i.e. the goodness of fit
    ylim ::NTuple{2, Float64} # limits for the predicted values

    function StandardGAM{N}(
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
function Base.show(io::IO, eq::StandardGAM{N}) where N
    npars = [length(coef) for coef in eq.β] |> sum

    @printf(
        io, 
        "StandardGAM{R^%d --> R} with %d terms, total %d parameters\n", 
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
function isclamped(f::StandardGAM{N}) where N
    return isfinite(f.ylim[1]) || isfinite(f.ylim[2])
end
# ------------------------------------------------------------------------------
function fit!(f::StandardGAM{N}, X::AbstractMatrix, y::AbstractVector) where N
    # expand: design matrix
    Greg = stack(f, X)

    # fit: least squares
    βvec = mvs.llsq(Greg, y, bias = false)

    # split: stacked coefficients to individual terms
    update!(f, βvec)

    # compute: pseudo R²
    ypred = Float64[f(x) for x in eachrow(X)]
    f.r2  = R²(y, ypred)

    return nothing
end
# ------------------------------------------------------------------------------
function StandardGAM(
    X     ::AbstractMatrix,
    y     ::AbstractVector,
    terms ::Vector{AbstractTerm} ;
    ylim  ::NTuple{2, Float64} = (-Inf, Inf),
)
    nobs, N = size(X)
    @assert length(y) == nobs "X and y must have the same number of obs/rows"
    @assert N > 0 "X must have at least one column"

    f = StandardGAM{N}(terms, ylim = ylim)

    fit!(f, X, y)

    return f
end
# ------------------------------------------------------------------------------
"""
    save_StandardGAM!(f::StandardGAM{N}, fpath::String)::Nothing where N

Save the GAM model `f` to a JSON file at `fpath`. The path should be valid and
refer to a file rather than a directory. Overwrite the file if it exists. For
aesthetic reason, the exported JSON file is human readable and pretty-printed.

## Example
```julia
X = rand(5000,2); Y = sum(X,dims = 2) |> vec
f = gam.StandardGAM(
    X, Y,
    [
        gam.Constant(),
        gam.Poly(degrees = [1,2]),
        gam.Logarithm(ϵ = 1E-8),
        gam.Exponential(),
        gam.CobbDouglas(D = 2),
    ]
)
gam.save_StandardGAM!(f, "trash.json")
```
"""
function save_StandardGAM!(f::StandardGAM{N}, fpath::String)::Nothing where N
    di = Dict{String, Any}(
        "type"   => "StandardGAM",
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
    load_StandardGAM(fpath::String)::StandardGAM

Load a StandardGAM model from a JSON file at `fpath`.

## Example
```julia
# run the example in `save_StandardGAM!` first
f2 = gam.load_StandardGAM("trash.json")
```
"""
function load_StandardGAM(fpath::String)::StandardGAM
    di = JSON3.read(fpath, Dict{String, Any})

    # assume: the type is always "StandardGAM"
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

    f = StandardGAM{N}(terms, ylim = ylim)
    for i in 1:m
        f.β[i] .= di["β"][i]
    end

    f.m    = m
    f.r2   = r2
    f.ylim = ylim

    return f
end
