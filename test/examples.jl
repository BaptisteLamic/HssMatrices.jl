include("../src/HssMatrices.jl")
using .HssMatrices
using LinearAlgebra
using AbstractTrees
using Plots

# test prrqr
# U = randn(100,3);
# V = randn(50,3);
# A = U * V';
# Q,R,p = HssMatrices.prrqr(A,1e-3);
# norm(A[:,p] - Q[:,1:size(R,1)]*R)

# generate Cauchy matrix
K(x,y) = (x-y) != 0 ? 1/(x-y) : 10000.
A = [ K(x,y) for x=-1:0.001:1, y=-1:0.001:1];

# test the simple implementation of cluster trees
m, n = size(A)
lsz = 64;
rcl = bisection_cluster(1:m, lsz)
ccl = bisection_cluster(1:n, lsz)

# test compression
hssA = hss_compress_direct(A, rcl, ccl);

@time hssA = hss_compress_direct(A, rcl, ccl);

# test recompression
# println("approximation error before recompression: ", norm(A - full(hssA)))
# hss_recompress!(hssA; tol=1e-1, reltol=false)
# println("approximation error after recompression: ", norm(A - full(hssA)))
# @time hss_recompress!(hssA; tol=1e-1, reltol=false)

# test mat-vec
x = randn(size(A,2), 3);
println("error in the matrix-vector products: ", norm(A*x - hssA*x))

# test the ULV based solver
b = randn(size(A,2), 5);
x = ulvfactsolve(hssA, b);
@time x = ulvfactsolve(hssA, b);
xcor = A\b;
println("error in the inversion: ", norm(x-xcor)/norm(xcor))


# # test computation of generators
# U1, V2 = generators(hssA, (1,2))
# A12 = A[1:hssA.m1,hssA.n1+1:end];
# println(norm(A12 - U1*hssA.B12*V2'))
# U2, V1 = generators(hssA, (2,1))
# A21 = A[hssA.m1+1:end,1:hssA.n1];|
# println(norm(A21 - U2*hssA.B21*V1'))


# #println(typeof(hssA)) 

# test the ULV based solver
# x = ulvfactsolve(hssA, b);

# @time x = ulvfactsolve(hssA, b);

# # test plotting
# plot = plotranks(hssA)

### TODO
# clean up the library Definitions
# implement other variants of ULV
# start work on the randomized compression