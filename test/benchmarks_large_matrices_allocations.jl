using Revise
using HssMatrices
using LinearAlgebra
using BenchmarkTools
using Random
using DataFrames
using SparseArrays
using Plots
Random.seed!(123)

function benchmark_full(A)
    return @benchmark full($A)
end

function benchmark_getindex(A)
    ii = randperm(100)
    jj = randperm(100)
    return @benchmark Aij = $A[$ii, $jj]
end

function benchmark_randcompress(A, rcl, ccl)
    return @benchmark randcompress_adaptive($A, $rcl, $ccl)
end

function benchmark_recompress(hssA)
    hssB = copy(hssA)
    return @benchmark hssA = recompress!($hssB; atol=1e-3, rtol=1e-3)
end

function benchmark_proper(hssA)
    hssB = hssA + hssA
    return @benchmark orthonormalize_generators!($hssB)
end

function benchmark_addition(hssA)
    return @benchmark hssC = $hssA + $hssA
end

function benchmark_multiplication(hssA)
    x = randn(size(hssA, 2), 10)
    return @benchmark y = $hssA * $x
end

function benchmark_ulvfactsolve(hssA)
    b = randn(size(hssA, 2), 10)
    return @benchmark x = ulvfactsolve($hssA, $b)
end

function benchmark_hssldivide(hssA, ccl)
    n = size(hssA, 1)
    hssX = randcompress_adaptive(1.0 * SparseMatrixCSC{Float64}(I, n, n), ccl, ccl, atol=1e-3, rtol=1e-3)
    return @benchmark hssC = ldiv!(copy($hssA), $hssX)
end

function construct_test_matrix()   # Define the test matrix and clusters
    invA = spdiagm((k => [Float64(k) for i in 1:10000-abs(k)] for k = -10:10)...)
    m, n = size(invA)
    lsz = 64
    rcl = bisection_cluster(1:m, leafsize=lsz)
    ccl = bisection_cluster(1:n, leafsize=lsz)
    hssInvA = randcompress_adaptive(invA, rcl, ccl)
    hssId = randcompress_adaptive(1.0 * SparseMatrixCSC{Float64}(I, n, n), ccl, ccl, atol=1e-6, rtol=1e-6)
    hssA = ldiv!(copy(hssInvA), hssId)
    return hssA, invA, rcl, ccl
end

# Function to run all benchmarks
function run_benchmarks(multithreaded, blas_threads)
    @show multithreaded
    @show blas_threads
    if multithreaded
        @show Threads.nthreads()
    end
    BLAS.set_num_threads(blas_threads)
    HssMatrices.setopts(multithreaded=multithreaded)
    hssA, invA, rcl, ccl = construct_test_matrix()

     # Run benchmarks
     results = Dict()
     println("Running benchmark: hssldivide")
     results[:hssldivide] = benchmark_hssldivide(hssA, ccl)

    
     return results
end
#Compute the performance for the multithreaded scenario using only 1 openBlas thread
hssA, invA, rcl, ccl = construct_test_matrix()
println("Warmup")
benchmark_hssldivide(hssA, ccl)
using Profile, PProf
Profile.Allocs.clear()
Profile.Allocs.@profile sample_rate=0.001 benchmark_hssldivide(hssA, ccl)
PProf.Allocs.pprof()
