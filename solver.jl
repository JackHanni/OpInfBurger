using MethodOfLines, ModelingToolkit, OrdinaryDiffEq, Plots, JLD2
using MethodOfLines: Interval   # no longer exported by ModelingToolkit
using OrdinaryDiffEqBDF: DFBDF  # DAE solver, no longer re-exported by OrdinaryDiffEq v7

# 1D viscous Burgers' equation:  u_t + u u_x = ν u_xx
#   x ∈ [0, 1],  t ∈ [0, 2]
#   u(x, 0) = sin(πx),  u(0, t) = u(1, t) = 0

@parameters x t
@variables u(..)
Dt = Differential(t)
Dx = Differential(x)
Dxx = Differential(x)^2

# bounds
x_min, t_min = 0.0, 0.0
x_max, t_max = 1.0, 2.0
# param
ν = 0.02                                  # small ν -> steep front, nonlinear term matters
# initial conditions
u0(x) = sin(π * x)

eq = Dt(u(x, t)) + u(x, t) * Dx(u(x, t)) ~ ν * Dxx(u(x, t))

bcs = [u(x, t_min) ~ u0(x),
       u(x_min, t) ~ 0.0,
       u(x_max, t) ~ 0.0]

domains = [x ∈ Interval(x_min, x_max),
           t ∈ Interval(t_min, t_max)]

@named pdesys = PDESystem(eq, bcs, domains, [x, t], [u(x, t)])

# Method of lines: finite differences in x, ODE solver in t
N = 201                                   # grid points (keep N ≳ 1/ν to resolve the front)
dx = (x_max - x_min) / (N - 1)
discretization = MOLFiniteDifference([x => dx], t)

prob = discretize(pdesys, discretization)   # PDE -> ODEProblem
sol = solve(prob, DFBDF(), saveat = 0.01)    # MOL returns a DAEProblem (BC nodes are algebraic) -> DAE solver

xs = sol[x]
ts = sol[t]
U = sol[u(x, t)]

# save solution data next to this script
datafile = joinpath(@__DIR__, "burgers_data.jld2")
jldsave(datafile; xs, ts, U, ν)
println("saved solution to ", datafile)

# static snapshot plot
idx = 1:20:length(ts)
plot(xs, U[:, idx], xlabel = "x", ylabel = "u", label = permutedims(["t = $(ts[i])" for i in idx]))
savefig(joinpath(@__DIR__, "burgers_sample_data.png"))

# animation
anim = @animate for k in eachindex(ts)
    plot(xs, U[:, k], xlabel = "x", ylabel = "u", ylims = (-0.05, 1.05), legend = false,
         title = "Burgers, ν = $(ν),  t = $(round(ts[k], digits = 2))")
end
gif(anim, joinpath(@__DIR__, "burgers.gif"), fps = 30)