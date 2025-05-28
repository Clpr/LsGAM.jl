lgam = include("src/LsGAM.jl")


tmp = let x0 = rand(3)
    # def: interaction term
    g = lgam.Interaction()

    # def: equivalent GAM
    f = lgam.StandardGAM{3}(
        lgam.AbstractTerm[
            lgam.Interaction(),
        ]
    )
    f.β[1] .= 1.0
    
    println("-"^80)
    # check: output
    [
        f(x0),
        g(x0) |> sum,
    ] |> println
    
    # check: gradient
    @time hcat(lgam.∂(g, x0), lgam.jacobian(g, x0)) |> display

    # check: Hessian
    println("-"^40)
    @btime lgam.∂2($g, $x0) |> sum
    println("-"^40)
    @btime lgam.hessian($f, $x0, Δ = 1E-6)
end










# simulated data
X = rand(100, 2)
y = sin.(X[:,1]) + cos.(X[:,2]) + randn(100) * 0.1

# define: blank model
eq = lgam.StandardGAM{2}(
    [
        lgam.Constant(),
        lgam.Poly(degrees = [1,2]),
        lgam.Logarithm(),
    ],
    ylim = (-Inf,Inf),
)

# fit: model to data
lgam.fit!(eq, X, y)
println(eq)

# define: directly fit
eq = lgam.StandardGAM(
    X, y,
    lgam.AbstractTerm[
        lgam.Constant(),
        lgam.Poly(degrees = [1,2]),
        lgam.Logarithm(),
        lgam.Interaction(),
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