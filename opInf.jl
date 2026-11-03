using LinearAlgebra, JLD2

datafile = joinpath(@__DIR__, "burgers_data.jld2")
@load datafile xs ts U ν
SVD = svd(U)
r = 15
sing_vals = SVD.S[1:r]
l_sing_vecs = SVD.U[:, 1:r]
X_hat = l_sing_vecs' * U
kron_vec(a, b) = vec([a[i] * b[j] for j in eachindex(b), i in eachindex(a)])
compact_kron(x) = [x[i] * x[j] for i in eachindex(x) for j in 1:i]
D = [X_hat' reduce(vcat, [compact_kron(X_hat_i)' for X_hat_i in eachcol(X_hat)])]