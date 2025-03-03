#using Revise
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
    println("Running benchmark: getindex")
    results[:getindex] = benchmark_getindex(hssA)
    println("Running benchmark: randcompress")
    results[:randcompress] = benchmark_randcompress(invA, rcl, ccl)
    println("Running benchmark: recompress")
    results[:recompress] = benchmark_recompress(hssA)
    println("Running benchmark: proper")
    results[:proper] = benchmark_proper(hssA)
    println("Running benchmark: addition")
    results[:addition] = benchmark_addition(hssA)
    println("Running benchmark: multiplication")
    results[:multiplication] = benchmark_multiplication(hssA)
    println("Running benchmark: ulvfactsolve")
    results[:ulvfactsolve] = benchmark_ulvfactsolve(hssA)
    println("Running benchmark: hssldivide")
    results[:hssldivide] = benchmark_hssldivide(hssA, ccl)

    return results
end

#First evaluate the nominal single-threaded performance
reference_results = run_benchmarks(false, Sys.CPU_THREADS)
#Compute the performance for the multithreaded scenario using only 1 openBlas thread
multithreaded_results = run_benchmarks(true, 1)
speedup_results = Dict()
for eachKey in keys(reference_results)
    speedup_results[eachKey] = median(reference_results[eachKey]).time / median(multithreaded_results[eachKey]).time 
end
speedup_plot = bar(
    string.(keys(speedup_results)),
    collect(values(speedup_results)),
    xlabel="Benchmark",
    ylabel="Speedup",
    title="Julia threads compared to OpenBLAS threads",
    size=(800, 600);  # Increase the size of the plot
    legend=false
)
savefig("speedup_plot.png")

hssA = construct_test_matrix()[1]
plt_right = plotranks(hssA)
savefig("hssranks_hssA.png")




