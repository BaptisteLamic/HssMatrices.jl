using TestItemRunner
using Revise
using HssMatrices
using LinearAlgebra

@testitem "compression of almost empty matrix" begin
    using LinearAlgebra
    leafsize = 32
    U = randn(leafsize, leafsize)
    V = randn(leafsize, leafsize)
    A = [U V; zero(U) U]
    B = [V U; zero(U) V]
    C = [zero(U) V; zero(V) zero(U)]
    D = A - C
    hssA = hss(A, leafsize = leafsize)
    hssB = hss(B, leafsize = leafsize)
    hssC = hss(C, leafsize = leafsize)
    hssD = hssA - hssC
    hssResult = hssB / hssD 
    c = 50
    testAccuracy(expected, result) = @test norm(expected - result)/norm(expected) ≤ c*HssOptions().rtol || 
    norm(expected - result) ≤ c*HssOptions().atol
    testAccuracy( full(hssB) / full(hssD),  hssResult )
end

@testitem "compression of almost empty matrix" begin
    using LinearAlgebra
    leafsize = 32
    U = randn(leafsize, leafsize)
    V = randn(leafsize, leafsize)
    A = [U V; zero(U) U]
   hssA = hss(A)
end

using LinearAlgebra
leafsize = 32
U = randn(leafsize, leafsize)
V = randn(leafsize, leafsize)
A = [U V; zero(U) U]
B = [V U; zero(U) V]
C = [zero(U) V; zero(V) zero(U)]
D = A - C
hssA = hss(A, leafsize = leafsize)
hssB = hss(B, leafsize = leafsize)
hssC = hss(C, leafsize = leafsize)
hssD = hssA - hssC
hssResult = hssB / hssA