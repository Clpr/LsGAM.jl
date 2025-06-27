#===============================================================================
INTERFACE METHODS:

ALL MODELS, ITS VARIANTS, AND ALL TERMS MUST BE COMPATIBLE WITH THESE METHODS.
===============================================================================#
export ∂, ∂2, jacobian, hessian
export coefficients, update!
export isclamped

#===============================================================================
SECTION: (VECTOR) TERMS g(x): R^n -> R^{k_i}

+ Notes:
    - all TERMS should have NO type parameters, i.e. `Term{N}` is not allowed.
    That information should be a field of the term if necessary.

    - all TERMS are not clamped while the whole GAM can be clamped by adding a
    clamping "layer" on top of the GAM.

    - all TERMS should be 2nd order differentiable, i.e. the Hessian is defined.
    
    - TERMS can have optional internal parameters e.g. `1/(x + Δ)` where `Δ` is
    a small constant to avoid division by zero. These parameters can be set
    during initialization of the term instance.
    
    - TERMS are not forced to hold in the whole real space. If an input `x` is
    outside the domain of the term, throw an error.
    
    - TERMS should be analytical functions. Their Jacobian and Hessian should
    also be analytical. Don't use numerical differentiation. This is for the
    sake of performance.
    
    - TERMS should be generically defined for arbitrary dimensionality `N` such
    that it is unnecessary declare `N` in the type parameter.

    - All TERMS should be able to be initialized with only dimensionality param
    `N` as the type parameter `g = Term()` should work. 
    
    - the constructors should NOT explicitly take the dimension of `x` but the
    value of `N` should be inferred from the length of `x`

    - TERMS should support serialization and deserialization, which implies that
    all fields of the terms should be serializable and supported by JSON syntax.
    A serialized term is a dictionary consisting of:
        - type name of the term, e.g. "name : Poly"
        - type parameters as a scalar or vector, e.g. "D : 3" for CobDouglas. 
          this is optional and only applies to specific terms that have type 
          params. leave it empty for terms without type params.
        - type fields as a dictionary, e.g. "coefficients : [1.0, 2.0, 3.0]"
          for terms that have fields. leave it empty for terms without fields.
    - Use "todict(term)" interface to convert a term to a serializable dict; use
      "fromdict(dict)" interface to convert a dict to a term instance. Each term
      type should implement these methods (`fromdict()` is special and we will
      explain the idea in the following bullet points).

    - It is encouraged to keep the field data structure of terms compatible with 
    the standard JSON to make life easier.

    - Because the type info is saved inside a serialized dict, so we maintenance
    a all-in-one-place implementation `fromdict(dict::Dict{String, Any})` in the
    `interface.jl` file. This method should be updated whenever a new term type
    is added.

    - For each term type, should implement a `fromdict_TYPENAME(dict)` method,
    where `TYPENAME` is the name of the term type, e.g. `fromdict_Poly(dict)`.
    This internal method will be called by the `fromdict(dict)` method using
    `getfield(LsGAM, Symbol("fromdict_", TYPENAME))(dict)`. This design allows
    parallel developement of new term types while keeping the `interface.jl`
    file tidy.

    - Test script for the serialization API:
    ```julia
    for tName in last.(split.(gam.AbstractTerm |> subtypes .|> string, "."))
       g = getfield(gam, Symbol(tName))()
       di = gam.todict(g)
       g2 = gam.fromdict(di)
       println(g, ": ", g == g2)
    end
    ````


+ Shared interfaces ------------------------------------------------------------

# callable evaluation: g(x): R^n -> R^{k_i}
(g::AbstractTerm)(x::AbstractVector)::Vector{Float64}

# gradient evaluation ∂g/∂x: R^n -> R^{k_i x n}
∂(g::AbstractTerm, x::AbstractVector)::Matrix{Float64}

# Hessian evaluation: ∂²g/∂x²:={∂²g[j]/∂x²}_{j=1}^{k_i}, i.e. k_i (n*n) matrices
∂2(g::AbstractTerm, x::AbstractVector)::Vector{Matrix{Float64}}

# length of the output vector given the dimensionality `n` of the input `x`
Base.length(g::AbstractTerm, n::Int)::Int

# numerical Jacobian of the term `g` at point `x`
jacobian(g::AbstractTerm, x::AbstractVector)::Vector{Float64}

# numerical Hessian of the term `g` at point `x`
hessian(g::AbstractTerm, x::AbstractVector)::Matrix{Float64}

# serialization & de-serialization API
todict(g::AbstractTerm)::Dict{String, Any}
fromdict(dict::Dict{String, Any})::AbstractTerm (specific to each term type)

===============================================================================#
function Base.show(io::IO, g::AbstractTerm)
    println(io, typeof(g))
    return nothing
end
# ------------------------------------------------------------------------------
"""
    jacobian(
        g::AbstractTerm,
        x::AbstractVector ;
        Δ::Float64 = 1e-6 # small perturbation for finite difference
    )::Vector{Float64}

Compute the numerical Jacobian of the term `g` at point `x`. This method is
used for terms where the analytical Jacobian is not defined or not available.
Returns a matrix of size `k_i x n`, where `k_i` is the output dimension of the
term `g` and `n` is the dimension of the input `x`.

Using central finite difference.
"""
function jacobian(
    g::AbstractTerm,
    x::AbstractVector ;
    Δ::Float64 = 1e-6 # small perturbation for finite difference
)::Matrix{Float64}
    n = length(x)
    k = length(g, n)
    J = zeros(Float64, k, n)
    for i in 1:n
        J[:, i] = (
            g(x + unitvec(n, i, v = Δ)) - g(x - unitvec(n, i, v = Δ))
        ) ./ (2.0 * Δ)
    end
    return J
end
# ------------------------------------------------------------------------------
"""
    hessian(
        g::AbstractTerm,
        x::AbstractVector ;
        Δ::Float64 = 1e-6 # small perturbation for finite difference
    )::Vector{Matrix{Float64}}

Compute the numerical Hessian of the term `g` at point `x`. This method is
used for terms where the analytical Hessian is not defined or not available.
Returns a `k`-vector of size `n*n` of matrices, where `k` is the output
dimension of the term `g` and `n` is the dimension of the input `x`.

Using central finite difference.

NOTE: the current implementation is VERY inefficient! Needs to be improved.
"""
function hessian(
    g::AbstractTerm,
    x::AbstractVector ;
    Δ::Float64 = 1e-6 # small perturbation for finite difference
)::Vector{Matrix{Float64}}
    n  = length(x)
    k  = length(g, n)
    Hs = [zeros(Float64, n, n) for _ in 1:k]
    for h in 1:k
        for i in 1:n, j in 1:n
            Hs[h][i, j] = (
                g(x + unitvec(n, i, v = Δ) + unitvec(n, j, v = Δ)) -
                g(x + unitvec(n, i, v = Δ) - unitvec(n, j, v = Δ)) -
                g(x - unitvec(n, i, v = Δ) + unitvec(n, j, v = Δ)) +
                g(x - unitvec(n, i, v = Δ) - unitvec(n, j, v = Δ))
            )[h] / (4.0 * Δ^2)
        end # i,j
    end # h
    return Hs
end
# ------------------------------------------------------------------------------
"""
    fromdict(di::Dict{String,Any})::AbstractTerm

De-serialize a term from a dictionary.

## Notes
- To avoid maintaining a huge if-else chain, we urge each term type to use a
single
- We implement the interface using meta which is not the very performant, as we 
expect the save/load operations would not be too often.

## I/O example
```julia
g = gam.ChebyshevT(degree = 3)
open("trash.json","w") do fio
    gam.JSON3.pretty(fio, g)
end
g3 = gam.JSON3.read("trash.json",Dict{String,Any}) |> gam.fromdict
```
"""
function fromdict(di::Dict{String,Any})::AbstractTerm
    return di |> getfield(
        LsGAM, 
        Symbol(
            :fromdict_, 
            di["type"]
        )
    )
end # fromdict





#===============================================================================
SECTION: GAM MODEL f(x) = β0 + ∑_{i=1}^m <g_i(x_i), β_i>

+ Notes:
    - GAMs can be clamped or not clamped. Clamping is done by adding an extra
    layer on top of an un-clamped GAM. The clamping layer forces the output to
    stary within a specified range, e.g. `y ∈ [ymin, ymax]`.
    - The domain of GAM should always be the whole real space, i.e. `x ∈ R^n`.
    - If a GAM is clamped, then for a point `x` where the output is clamped, the
    analytical Jacobian and Hessian of the GAM are not defined. The methods of 
    derivatives `∂(g::AbstractGAM)` and `∂2(g::AbstractGAM)` then use numerical
    differentiation to compute the Jacobian and Hessian.
    - Implements numerical differentiation for the Jacobian and Hessian for the
    clamped GAMs. Finite difference for now, may be replaced with more
    sophisticated methods later.
    - The bias/intercept term should be manually added to the GAM but not be
    controlled using an arguments like `bias = true`.
    - the constructors should NOT explicitly take the dimension of `x` but the
    value of `N` should be inferred from the length of `x`

    - One should not be able to modify the GAM specification after it has been
    constructed. This leads to incompatibility with the fitted model.

    - Each GAM should have a `save!(f::AbstractGAM, path::String)` method to 
    export the GAM to a JSON file, and a `load(path::String)::AbstractGAM`
    method to load the GAM from a JSON file. The JSON file should corresponds to
    a generic dict that contains the following fields:
        - `type`: the type of the GAM, e.g. "StandardGAM"
        - `type_N`: the dimension of the input `x`, e.g. 3 for `StandardGAM{3}`
        - `terms`: a vector of serialized terms, each term is a dict that is
          created by the `todict(term)` method.
        - `β`: a vector of vectors, each vector corresponds to the coefficients
          of the corresponding term in `terms`. The coefficients are stored as
          vectors of floats. The length of each vector should match the output
          dimension of the corresponding term.
        - other fields of the GAM which is implemented by each GAM type. They
          are handled by the `save_TYPENAME!(f, fpath)`, `load_TYPENAME(fpath)`
          methods.

    - It is encouraged to keep the field data structure of GAMs compatible with 
    the standard JSON to make life easier.

    - Each GAM type should ONLY have a single integer type parameter `N` which
    indicates the dimension of the input `x`. This is to infer the loaded model
    structure from the JSON file. No more type parameters are allowed due to the
    aesthetic reason.

    - Like the I/O methods for the terms, each GAM type should implement a
    `save_TYPENAME!(f,fpath)` and `load_TYPENAME(fpath)` methods, where
    `TYPENAME` is the name of the GAM type, e.g. `save_StandardGAM!`,
    `load_StandardGAM`. These methods will be called by the `save!` and `load`
    interface methods in the `interface.jl` file.



+ Shared interfaces ------------------------------------------------------------

# must have type parameters:
- N::Int # dimension of input x, e.g. Constant{N} <: AbstractTerm{N}

# must-have fields:
- terms::Vector{AbstractTerm} # vector of terms g_i(x_i)
- β    ::Vector{Vector{Float64}} # vector of coefficients β_i for each term g_i
- m    ::Int # number of terms g_i
- r2   ::Float64 # pseudo R² of the model, i.e. the goodness of fit


# callable evaluation: f(x): R^n -> R
(f::AbstractGAM{N})(x::AbstractVector)::Float64

# gradient evaluation ∂f/∂x: R^n -> R^{n}
∂(f::AbstractGAM{N}, x::AbstractVector)::Vector{Float64}

# Hessian evaluation: ∂²f/∂x²: R^n -> R^{n x n}
∂2(f::AbstractGAM{N}, x::AbstractVector)::Matrix{Float64}

# stack/vectorize the coefficients β_i into a single vector
coefficients(f::AbstractGAM{N})::Vector{Float64}

# create a long vector by stacking all `g_i(x)` outputs given a point `x`
Base.stack(f::AbstractGAM{N}, x::AbstractVector)::Vector{Float64}
Base.stack(f::AbstractGAM{N}, X::AbstractMatrix)::Matrix{Float64} # X (#obs * n)

# update the coefficients of the GAM in-place from a stacked vector
update!(f::AbstractGAM{N}, βvec::AbstractVector) where N

# dimension of the input `x` vector
Base.ndims(f::AbstractGAM{N})::Int

--------------------------------------------------------------------------------
(The above methods can be implemented here since all terms are callable)
--------------------------------------------------------------------------------

# show
Base.show(io::IO, f::AbstractGAM{N})

# check if the GAM is clamped
isclamped(f::AbstractGAM)::Bool

# construct with default blank coefficients
GAM_NAME{N}(
    terms::Vector{AbstractTerm} ;
    ylim ::NTuple{2,Float64} = (-Inf, Inf), # for clamped GAMs, y-limits
)

# fitting the GAM to data (constructor)
fit!(f::AbstractGAM{N}, X::Matrix{Float64}, y::Vector{Float64})

# construct & fit (the dimensionality is inferred from X)
GAM_NAME(
    X::Matrix{Float64}, # size: #obs x n 
    y::Vector{Float64}, # size: #obs
    terms::Vector{AbstractTerm} ;
    ylim::NTuple{2,Float64} = (-Inf, Inf), # for clamped GAMs
)

# save & load (serialization)
save!(f::AbstractGAM{N}, path::String)::Nothing
load(path::String)::AbstractGAM{N}


===============================================================================#
function Base.show(io::IO, f::AbstractGAM{N}) where N
    @printf(io, "%s(n = %d, m = %d, r2 = %.4f)", typeof(f), N, f.m, f.r2)
    return nothing
end
# ------------------------------------------------------------------------------
function Base.size(f::AbstractGAM{N})::NTuple{2,Int} where N
    return (N, f.m)
end
# ------------------------------------------------------------------------------
function (f::AbstractGAM{N})(x::AbstractVector)::Float64 where N
    @assert length(x) == N "x must have length $(N) but got $(length(x))"
    y::Float64 = 0.0
    for i in 1:f.m
        y += sum(f.terms[i](x) .* f.β[i]) # <(ki*1),(ki*1)> => (1*1)
    end
    return isclamped(f) ? clamp(y, f.ylim...) : y
end
# ------------------------------------------------------------------------------
"""
    ∂(f::AbstractGAM, x::AbstractVector)::Vector{Float64}

Compute the analytical gradient of the GAM `f` at point `x`. Chain rule applies.
If the GAM is clamped, this method will throw an error.

Returns a vector of size `n`, where `n` is the dimension of the input `x`.
"""
function ∂(f::AbstractGAM{N}, x::AbstractVector)::Vector{Float64} where N
    if isclamped(f)
        return jacobian(f, x) # use numerical differentiation for clamped GAMs
    end

    ∇ = zeros(Float64, N)

    for i in 1:f.m
        ∇ .+= ∂(f.terms[i], x)' * f.β[i] # (ki*n)' * (ki*1) = (n*1)
    end

    return ∇
end
# ------------------------------------------------------------------------------
"""
    ∂2(f::AbstractGAM, x::AbstractVector)::Matrix{Float64}

Compute the analytical Hessian of the GAM `f` at point `x`. Chain rule applies.
If the GAM is clamped, this method will throw an error.
Returns a matrix of size `n x n`, where `n` is the dimension of the input `x`.
"""
function ∂2(f::AbstractGAM{N}, x::AbstractVector)::Matrix{Float64} where N
    if isclamped(f)
        return hessian(f, x) # use numerical differentiation for clamped GAMs
    end

    H = zeros(Float64, N, N)
    for i in 1:f.m
        H += sum(f.β[i] .* ∂2(f.terms[i], x))
    end # i
    return H
end
# ------------------------------------------------------------------------------
"""
    jacobian(f::AbstractGAM, x::AbstractVector)::Vector{Float64}

Compute the numerical Jacobian of the GAM `f` at point `x`. This method is used
for clamped GAMs where the analytical Jacobian is not defined due to clamping.
Returns a vector of size `n`, where `n` is the dimension of the input `x`.

Using central finite difference.
"""
function jacobian(
    f::AbstractGAM{N}, 
    x::AbstractVector ;
    Δ::Float64 = 1e-6 # small perturbation for finite difference
)::Vector{Float64} where N
    @assert length(x) == N "x must have length $(N) but got $(length(x))"

    J = zeros(Float64, N)
    for i in 1:N
        J[i] = (
            f(x + unitvec(N, i, v = Δ)) - f(x - unitvec(N, i, v = Δ))
        ) / (2.0 * Δ)
    end
    return J
end
# ------------------------------------------------------------------------------
"""
    hessian(f::AbstractGAM, x::AbstractVector)::Matrix{Float64}

Compute the numerical Hessian of the GAM `f` at point `x`. This method is used
for clamped GAMs where the analytical Hessian is not defined due to clamping.
Returns a matrix of size `n x n`, where `n` is the dimension of the input `x`.

Using central finite difference.
"""
function hessian(
    f::AbstractGAM{N},
    x::AbstractVector ;
    Δ::Float64 = 1e-6 # small perturbation for finite difference
)::Matrix{Float64} where N
    @assert length(x) == N "x must have length $(N) but got $(length(x))"

    H = zeros(Float64, N, N)
    for i in 1:N
        for j in 1:N
            H[i, j] = (
                f(x + unitvec(N, i, v = Δ) + unitvec(N, j, v = Δ)) -
                f(x + unitvec(N, i, v = Δ) - unitvec(N, j, v = Δ)) -
                f(x - unitvec(N, i, v = Δ) + unitvec(N, j, v = Δ)) +
                f(x - unitvec(N, i, v = Δ) - unitvec(N, j, v = Δ))
            ) / (4.0 * Δ^2)
        end
    end
    return H
end
# ------------------------------------------------------------------------------
"""
    coefficients(f::AbstractGAM)::Vector{Float64}

Stack the coefficients of the GAM `f` into a single vector. This is useful for
optimizing the GAM or for extracting the coefficients for further analysis.
The coefficients are stacked in the order of the terms, i.e. the first `k_1`
coefficients correspond to the first term, the next `k_2` coefficients to the
second term, and so on.
"""
function coefficients(f::AbstractGAM{N})::Vector{Float64} where N
    return vcat(f.β...)
end
# ------------------------------------------------------------------------------
"""
    Base.stack(f::AbstractGAM{N}, x::AbstractVector)::Vector{Float64} where N

Stack the outputs of the GAM `f` for the input vector `x`. This is useful for
exporting the design matrix of the GAM, where each term's output is stacked
into a single vector. The output is a vector of size `k_1 + k_2 + ... + k_m`,
where `k_i` is the output dimension of the `i-th` term `g_i(x_i)`.
"""
function Base.stack(
    f::AbstractGAM{N},
    x::AbstractVector
)::Vector{Float64} where N
    return reduce(vcat, [f.terms[i](x) for i in 1:f.m])
end
# ------------------------------------------------------------------------------
"""
    Base.stack(f::AbstractGAM{N}, X::AbstractMatrix)::Matrix{Float64} where N

Stack the outputs of the GAM `f` for each row of the input matrix `X`. This in
fact exports the design matrix of the GAM.
"""
function Base.stack(
    f::AbstractGAM{N},
    X::AbstractMatrix
)::Matrix{Float64} where N
    @assert size(X, 2) == N "X must have $(N) columns but got $(size(X, 2))"
    return reduce(
        hcat,
        [stack(f, x) for x in eachrow(X)] # stack each row
    ) |> permutedims
end
# ------------------------------------------------------------------------------
"""
    update!(f::AbstractGAM{N}, βvec::AbstractVector)::Nothing where N

Update the coefficients of the GAM `f` in-place from a stacked vector `βvec`.
The vector `βvec` should have the same length as the total number of coefficient
vectors in `f`. This method is useful in fitting the GAM to data or working with
the design matrix directly.
"""
function update!(
    f   ::AbstractGAM{N},
    βvec::AbstractVector
) where N
    Ks = Int[length(term, N) for term in f.terms]
    Nβ = length(βvec)
    @assert Nβ == sum(Ks) "require $(sum(Ks)) elements but got $(Nβ)"
    ctr = 1
    for i in 1:f.m
        f.β[i] .= βvec[ctr:ctr + Ks[i] - 1]
        ctr += Ks[i]
    end
    return nothing
end
# ------------------------------------------------------------------------------
function Base.ndims(f::AbstractGAM{N})::Int where N
    return N
end
# ------------------------------------------------------------------------------
"""
    save!(f::AbstractGAM{N}, fpath::String)::Nothing where N

Save the GAM model `f` to a JSON file at `fpath`. The path should be valid and
the file should be writable.
"""
function save!(f::AbstractGAM{N}, fpath::String)::Nothing where N
    gamType = split(f |> typeof |> string, ".")[end]
    gamType = split(gamType, "{")[1] # remove type parameters if any

    getfield(
        LsGAM, 
        Symbol(
            "save_", 
            gamType,
            "!"
        )
    )(f, fpath)
    return nothing
end
# ------------------------------------------------------------------------------
"""
    load(fpath::String)::AbstractGAM{N} where N

Load a GAM model from a JSON file at `fpath`.
"""
function load(fpath::String)
    di      = JSON3.read(fpath, Dict{String, Any})
    gamType = di["type"]
    
    return getfield(
        LsGAM,
        Symbol(
            "load_",
            gamType,
        )
    )(fpath)
end
