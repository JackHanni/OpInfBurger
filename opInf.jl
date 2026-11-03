using LinearAlgebra, JLD2
include("burgers_fd.jl")

datafile = joinpath(@__DIR__, "burgers_data.jld2")
@load datafile xs ts U ν
SVD = svd(U)
r = 15
l_sing_vecs = SVD.U[:, 1:r]
X_hat = l_sing_vecs' * U
kron_vec(a, b) = vec([a[i] * b[j] for j in eachindex(b), i in eachindex(a)])
compact_kron(x) = [x[i] * x[j] for i in eachindex(x) for j in 1:i]
D = [X_hat' reduce(vcat, [compact_kron(X_hat_i)' for X_hat_i in eachcol(X_hat)])]
u_dot = reduce(hcat, [burgers_rhs!(similar(U[:, i]), U[:, i], ν, xs[2] - xs[1]) for i in axes(U, 2)])
Y = (l_sing_vecs' * u_dot)'
λ = 1e-4
O = ( (D'*D+λ*I(size(D,2))) \ (D'*Y) )'
A_hat = O[:, 1:r]
H_hat = O[:, r+1:end]

rom_rhs!(dq, q) = (dq .= A_hat * q .+ H_hat * compact_kron(q))
q = copy(X_hat[:, 1])
k1, k2, k3, k4, tmp = (similar(q) for _ in 1:5)
Q = zeros(r, length(ts))
Q[:, 1] .= q
sub = 10
dt = (ts[2] - ts[1]) / sub
for n in 1:length(ts)-1
    for _ in 1:sub
        rk4_step!(q, dt, rom_rhs!, k1, k2, k3, k4, tmp)
    end
    Q[:, n+1] .= q
end
U_rom = l_sing_vecs * Q
println("relative error: ", norm(U_rom - U) / norm(U))
lo, hi = extrema(U)
pad = 0.05 * max(hi - lo, eps())
res_inf = [norm(U[:, k] - U_rom[:, k], Inf) for k in eachindex(ts)]
anim = @animate for k in eachindex(ts)
    plot(xs, U[:, k], label = "full model", xlabel = "x", ylabel = "u",
         ylims = (lo - pad, hi + pad),
         title = "Burgers, r = $r,  t = $(round(ts[k], digits = 2)),  ‖res‖∞ = $(round(res_inf[k], sigdigits = 3))")
    plot!(xs, U_rom[:, k], label = "OpInf ROM", linestyle = :dash)
end
gif(anim, joinpath(@__DIR__, "burgers_rom.gif"), fps = 30)
