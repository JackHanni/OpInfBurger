using LinearAlgebra, JLD2

datafile = joinpath(@__DIR__, "burgers_data.jld2")
@load datafile xs ts U ν
SVD = svd(U)
r = 10
sing_vals = SVD.S[1:r]
l_sing_vecs = SVD.U[:, 1:r]
X_hat = l_sing_vecs' * U          # project snapshots onto POD basis: r × n_snapshots
