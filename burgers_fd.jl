# 1D viscous Burgers' equation, hand-written finite differences
#
#   u_t + u u_x = ν u_xx,   x ∈ [x_min, x_max],  t ∈ [0, t_max]
#   u(x, 0) = u0(x),   u(x_min, t) = uL,   u(x_max, t) = uR
#
# Space: 2nd-order centered differences.  Time: explicit RK4.
# Change the problem by editing `default_params` below, or from another script:
#
#   include("burgers_fd.jl")
#   out = run_burgers(merge(default_params, (ν = 0.01, N = 401, name = "nu0p01")))
#
# Run directly with:  julia burgers_fd.jl

using JLD2, Plots

# ------------------------------------------------------------------ parameters
const default_params = (
    ν       = 0.02,                 # viscosity
    x_min   = 0.0,
    x_max   = 1.0,
    N       = 201,                  # grid points incl. boundaries (keep N ≳ 1/ν)
    t_max   = 2.0,                  # final time
    save_dt = 0.01,                 # snapshot spacing
    u0      = x -> sin(π * x),      # initial condition
    uL      = 0.0,                  # Dirichlet BC at x_min
    uR      = 0.0,                  # Dirichlet BC at x_max
    cfl     = 0.4,                  # safety factor on the explicit time-step limit
    outdir  = @__DIR__,             # where outputs go
    name    = "burgers",            # output file prefix
    make_plot = true,
    make_gif  = true,
    gif_fps   = 30,
)

# ------------------------------------------------------------------ spatial operator
"""
    burgers_rhs!(du, u, ν, dx)

Semi-discrete right-hand side du/dt at interior nodes. Boundary conditions are
Dirichlet, so du = 0 there.
"""
function burgers_rhs!(du, u, ν, dx)
    N = length(u)
    du[1] = 0.0
    du[N] = 0.0
    @inbounds for i in 2:N-1
        adv  = u[i] * (u[i+1] - u[i-1]) / (2dx)
        diff = ν * (u[i+1] - 2u[i] + u[i-1]) / dx^2
        du[i] = -adv + diff
    end
    return du
end

# ------------------------------------------------------------------ time integration
function rk4_step!(u, dt, f!, k1, k2, k3, k4, tmp)
    f!(k1, u);  @. tmp = u + dt/2 * k1
    f!(k2, tmp); @. tmp = u + dt/2 * k2
    f!(k3, tmp); @. tmp = u + dt   * k3
    f!(k4, tmp); @. u  += dt/6 * (k1 + 2k2 + 2k3 + k4)
    return u
end

"""
    solve_burgers(p) -> (xs, ts, U, dUdt, dt)

U[:, k] is the solution at ts[k]; dUdt[:, k] is the exact semi-discrete time
derivative there.
"""
function solve_burgers(p)
    xs = collect(range(p.x_min, p.x_max, length = p.N))
    dx = xs[2] - xs[1]

    u = p.u0.(xs)
    u[1], u[end] = p.uL, p.uR

    # explicit stability: diffusion dt ≤ dx²/(2ν), advection dt ≤ dx/max|u|.
    # For Burgers with Dirichlet BCs, max|u| never exceeds its initial/boundary value.
    umax = max(maximum(abs, u), abs(p.uL), abs(p.uR), eps())
    dt_lim = min(dx^2 / (2p.ν), dx / umax)

    # cell Péclet check: centered advection oscillates if umax*dx/ν > 2
    Pe = umax * dx / p.ν
    Pe > 2 && @warn "Cell Péclet number $(round(Pe, digits=2)) > 2: expect oscillations near steep fronts. Increase N (≳ $(ceil(Int, umax*(p.x_max-p.x_min)/(2p.ν)) + 1))."

    nsave = round(Int, p.t_max / p.save_dt)
    isapprox(nsave * p.save_dt, p.t_max; rtol = 1e-10) ||
        error("t_max must be a multiple of save_dt")
    sub = ceil(Int, p.save_dt / (p.cfl * dt_lim))      # RK4 steps per snapshot
    dt  = p.save_dt / sub

    ts   = collect(range(0.0, p.t_max, length = nsave + 1))
    U    = zeros(p.N, nsave + 1)
    dUdt = zeros(p.N, nsave + 1)
    f!(du, v) = burgers_rhs!(du, v, p.ν, dx)
    k1, k2, k3, k4, tmp = (similar(u) for _ in 1:5)

    U[:, 1] .= u
    f!(view(dUdt, :, 1), u)
    for n in 1:nsave
        for _ in 1:sub
            rk4_step!(u, dt, f!, k1, k2, k3, k4, tmp)
        end
        all(isfinite, u) || error("solution blew up at t = $(ts[n+1]); reduce cfl")
        U[:, n+1] .= u
        f!(view(dUdt, :, n+1), u)
    end
    return xs, ts, U, dUdt, dt
end

# ------------------------------------------------------------------ outputs
function save_outputs(p, xs, ts, U, dUdt, dt)
    mkpath(p.outdir)
    datafile = joinpath(p.outdir, "$(p.name)_data.jld2")
    jldsave(datafile; xs, ts, U, dUdt,
            ν = p.ν, N = p.N, dt, save_dt = p.save_dt, uL = p.uL, uR = p.uR)
    println("saved data      → ", datafile)

    if p.make_plot
        idx = round.(Int, range(1, length(ts), length = 11))
        plt = plot(xs, U[:, idx], xlabel = "x", ylabel = "u",
                   label = permutedims(["t = $(round(ts[i], digits = 2))" for i in idx]))
        pngfile = joinpath(p.outdir, "$(p.name)_snapshots.png")
        savefig(plt, pngfile)
        println("saved plot      → ", pngfile)
    end

    if p.make_gif
        lo, hi = extrema(U)
        pad = 0.05 * max(hi - lo, eps())
        anim = @animate for k in eachindex(ts)
            plot(xs, U[:, k], xlabel = "x", ylabel = "u", legend = false,
                 ylims = (lo - pad, hi + pad),
                 title = "Burgers, ν = $(p.ν),  t = $(round(ts[k], digits = 2))")
        end
        giffile = joinpath(p.outdir, "$(p.name).gif")
        gif(anim, giffile, fps = p.gif_fps)
    end
    return datafile
end

"""
    run_burgers(p = default_params)

Solve, save data/plot/gif, and return a NamedTuple of results.
"""
function run_burgers(p = default_params)
    t_solve = @elapsed (xs, ts, U, dUdt, dt) = solve_burgers(p)
    println("solved: N = $(p.N), ν = $(p.ν), dt = $(round(dt, sigdigits = 3)), ",
            "$(length(ts)) snapshots in $(round(t_solve, digits = 3)) s")
    save_outputs(p, xs, ts, U, dUdt, dt)
    return (; xs, ts, U, dUdt, dt)
end

# run only when executed as a script, not when include()d
if abspath(PROGRAM_FILE) == @__FILE__
    run_burgers()
end
