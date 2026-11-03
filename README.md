# OpInfBurger

Operator Inference (OpInf) reduced-order models for the 1D viscous Burgers' equation

$$
u_t + u\,u_x = \nu\,u_{xx}, \qquad x \in [0, 1],\; t \in [0, T],
$$

with $u(x, 0) = \sin(\pi x)$ and $u(0, t) = u(1, t) = 0$ by default.

Burgers' nonlinearity is exactly quadratic, so the OpInf model form
$\dot{\hat q} = \hat A \hat q + \hat H (\hat q \otimes \hat q)$ matches the structure of the
true POD-Galerkin reduced model.

## Files

| File | Purpose |
|---|---|
| `burgers_fd.jl` | Full-order solver: 2nd-order centered finite differences in space, RK4 in time. Saves snapshots, exact time derivatives, a plot, and a GIF. |
| `opInf.jl` | Operator inference: POD basis from the snapshots, projection to reduced coordinates, and the linear + quadratic data matrix (work in progress). |
| `solver.jl` | The same problem solved with MethodOfLines.jl / ModelingToolkit.jl (much slower to load; kept for comparison). |

## Requirements

Julia ≥ 1.10 with:

```julia
using Pkg; Pkg.add(["JLD2", "Plots"])
```

`solver.jl` additionally needs `MethodOfLines`, `ModelingToolkit`, `OrdinaryDiffEq`, and `OrdinaryDiffEqBDF`.

## Usage

The snapshot data and figures are already in the repo, so `opInf.jl` runs straight after cloning.
To regenerate them (takes a few seconds; most of it is plotting):

```bash
julia burgers_fd.jl
```

This writes, next to the script:

- `burgers_data.jld2` — `xs`, `ts`, `U` (N × n_snapshots), `dUdt` (exact semi-discrete time derivatives), `ν`, `N`, `dt`, `save_dt`, `uL`, `uR`
- `burgers_snapshots.png` — solution at 11 times
- `burgers.gif` — animation

`burgers_sample_data.png` is the corresponding plot from `solver.jl`.

Then run the reduced model script:

```bash
julia opInf.jl
```

### Changing the problem

All settings live in `default_params` at the top of `burgers_fd.jl` (ν, domain, grid size `N`,
final time, snapshot spacing, initial condition, boundary values, output name, plot/GIF switches).
The time step is chosen automatically from the diffusion and advection stability limits, and the
solver warns if the grid is too coarse for ν (cell Péclet number > 2).

To run a variant without editing the file:

```julia
include("burgers_fd.jl")
out = run_burgers(merge(default_params, (ν = 0.01, N = 401, name = "nu0p01")))
```

The `name` prefix keeps outputs from different runs separate.

## Notes

- Keep `N ≳ 1/ν`; smaller ν gives steeper fronts and needs a finer grid.
- Use the saved `dUdt` rather than finite differences of `U` when fitting OpInf operators:
  accurate time derivatives make the regression much better conditioned.
- Generated data and figures are tracked in git. If you rerun `burgers_fd.jl` with new
  parameters, commit the updated `burgers_data.jld2` and figures so they stay in sync with the code.
