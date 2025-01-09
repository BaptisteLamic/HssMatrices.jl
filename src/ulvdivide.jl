### Apply the inverse of an HSS matrix to another HSS matrix
#
# as seen in
# Chandrasekaran, S., Gu, M., & Pals, T. (2006). A Fast $ULV$ Decomposition Solver for Hierarchically Semiseparable Representations.
# SIAM Journal on Matrix Analysis and Applications, 28(3), 603–622. https://doi.org/10.1137/S0895479803436652
#
# Massei, S., Robol, L., & Kressner, D. (n.d.). hm-toolbox: MATLAB SOFTWARE FOR HODLR AND HSS MATRICES.
# Retrieved from http://scg.ece.ucsb.edu/software.html
#
# Written by Boris Bonev, Feb. 2021

# function for acess that imitate the behavior in LinearAlgebra.jl
ldiv!(hssA::HssMatrix, hssB::HssMatrix) = _ldiv!(copy(hssA), hssB)
# lazy implementation of rdiv!
function rdiv!(hssA::HssMatrix, hssB::HssMatrix)
  hssA = adjoint(_ldiv!(adjoint(hssB), adjoint(hssA)))
end

## ULV divide algorithm to apply the inverse to another HSS matrix
# temporary name for function that actually just computes the ULV factorization
# the cluster structure in hssA and hssB should be compatible w/e that means...
function _ldiv!(hssA::HssMatrix, hssB::HssMatrix)
  if isleaf(hssA)
    D, U, V = _hssleaf(hssB)
    D = full(hssA) \ D
    return HssMatrix(D, U, V, isroot(hssB))
  else
    # bottom-up stage of the ULV solution algorithm
    hssL, QU, QL, QV, mk, nk, ktree  = _ulvfactor_leaves!(hssA, 0)
    hssB = _utransforms!(hssB, QU)
    hssQB = _extract_crows(hssB, nk)
    hssY0 = _ltransforms!(hssB, QL)

    # early exit if all remaining blocks are 0.
    if size(hssQB,1) == 0; return hssB = _vtransforms!(hssB, QV); end
    # TODO: this still fails if only one hss block gets fully eliminated - fix that!

    hssQB = hssQB - hssL * hssY0 # multiply triangularized part with solved part and substract from the
    hssQB = recompress!(hssQB)
    hssQB = prune_leaves!(hssQB)

    # reduce to the remainder block (and regain sqare HSS matrix for recursive division)
    hssL = _extract_ccols(hssL, nk) # extract uncompressed rows to form Matrix with one less level
    hssL = prune_leaves!(hssL)

    # recursively call mldivide
    hssY1 = _ldiv!(hssL, hssQB)

    # do the unpacking
    hssB = _unpackadd_rows!(hssY0, hssY1, ktree)
    hssB = _vtransforms!(hssB, QV)
    hssB = recompress!(hssB)
  end
end

## The core routines which make up the division follow here
# core routine to reduce the rows and triangularize the diagonal block via QR/LQ decompositions
function _ulvreduce!(D::Matrix{T}, U::Matrix{T}, V::Matrix{T}) where T
  T <: Complex ? adj = 'C' : adj = 'T'
  m, n = size(D)
  k = size(U, 2)
  nk = min(m-k,n)
  ind = 1:m-k
  cind = m-k+1:m
  # can't be compressed, exit early
  if k >= m
    #@warn "Encountered a full-rank block with k=$(k). Clustering might not yield best performance!"
    L1 = Matrix{T}(undef, 0, 0)
    qlf = (Matrix{T}(undef, m, 0), Vector{T}(undef, 0))
    lqf = (Matrix{T}(undef, 0, n), Vector{T}(undef, 0))
  else
    # form QL decomposition of the row generators and apply it
    qlf = geqlf!(U);
    U = tril(U[end-k+1:end,:]) # k x k block
    ormql!('L', adj, qlf..., D) # transform the diagonal block
    # Form the LQ decomposition of the first m-k rows of D
    lqf = gelqf!(D[1:end-k,:])
    L1 = tril(lqf[1])
    L1 = L1[:,1:nk]
    L2 = ormlq!('R', adj, lqf..., D[end-k+1:end,:]) # update the bottom block of the diagonal block
    V = ormlq!('L', 'N', lqf..., V) # compute the updated off-diagonal generators on the right
    D = L2
  end
  return D, U, V, L1, qlf, lqf, m-k, nk, k
end

# recursive function that operates only on the leaves
function _ulvfactor_leaves!(hssA::HssMatrix{T}, co::Int) where T
  if isleaf(hssA)
    cols = collect(co .+ (1:size(hssA,2)))
    m, n = size(hssA.D)
    D, U, V, L1, qlf, lqf, mk, nk, k = _ulvreduce!(hssA.D, hssA.U, hssA.V)
    # remember which columns got reduced
    QU = BinaryNode(qlf)
    QL = BinaryNode(L1)
    QV = BinaryNode(lqf) # check whether indices are correct
    return HssMatrix(D, U, V, hssA.rootnode), QU, QL, QV, BinaryNode(mk), BinaryNode(nk), BinaryNode(k)
  else
    m1, n1 = hssA.sz1; m2, n2 = hssA.sz2
    task = RecursionTools.spawn(_ulvfactor_leaves!,(hssA.A11, co), true)
    hssA.A22, QU2, QL2, QV2, mk2, nk2, k2 = _ulvfactor_leaves!(hssA.A22, co+n1)
    hssA.A11, QU1, QL1, QV1, mk1, nk1, k1 = RecursionTools.fetch(task)
    # update sizes
    hssA.sz1 = size(hssA.A11); hssA.sz2 = size(hssA.A22)
    QU = BinaryNode(QU1, QU2)
    QL = BinaryNode(QL1, QL2)
    QV = BinaryNode(QV1, QV2)
    mk = BinaryNode(mk1, mk2)
    nk = BinaryNode(nk1, nk2)
    ktree = BinaryNode(k1, k2)
    return hssA, QU, QL, QV, mk, nk, ktree # not sure mk, nk are needed
  end
end

## functions to apply U, L and V transforms at the leaf level to hssB
function _utransforms_kernel!(hssA::HssMatrix, Q::BinaryNode)
  eltype(hssA) <: Complex ? adj = 'C' : adj = 'T'
  if size(Q.data[1], 1) == 0 return hssA end
  hssA.D = ormql!('L', adj, Q.data..., hssA.D)
  hssA.U = ormql!('L', adj, Q.data..., hssA.U)
  return hssA
end
function _ltransforms_kernel!(hssA::HssMatrix, Q::BinaryNode)
  nk = size(Q.data,1)
  hssA.D[1:nk, :] = trsm('L', 'L', 'N', 'N', eltype(hssA)(1), Q.data, hssA.D[1:nk,:])
  hssA.U[1:nk, :] = trsm('L', 'L', 'N', 'N', eltype(hssA)(1), Q.data, hssA.U[1:nk,:])
  hssA.D[nk+1:end, :] .= eltype(hssA)(0)
  hssA.U[nk+1:end, :] .= eltype(hssA)(0)
  return hssA
end
function _vtransforms_kernel!(hssA::HssMatrix, Q::BinaryNode)
  eltype(hssA) <: Complex ? adj = 'C' : adj = 'T' 
  if size(Q.data[1], 1) == 0 return hssA end
  hssA.D = ormlq!('L', adj, Q.data..., hssA.D)
  hssA.U = ormlq!('L', adj, Q.data..., hssA.U)
  return hssA
end
# function _rapplyv!(hssA::HssMatrix, Q::BinaryNode)
#   eltype(hssA) <: Complex ? adj = 'C' : adj = 'T'
#   hssA.D = ormlq!('R', adj, Q.data..., hssA.D)
#   hssA.V = ormlq!('R', adj, Q.data..., hssA.U)
#   return hssA
# end

# generate branch level recursive function for each transform
for (f,g) in zip((:_utransforms!, :_ltransforms!, :_vtransforms!), (:_utransforms_kernel!, :_ltransforms_kernel!, :_vtransforms_kernel!))
  @eval begin
    function $f(hssA::HssMatrix, Qtree::BinaryNode)
      if isleaf(hssA)
        hssA = $g(hssA, Qtree)
      else
        task = RecursionTools.spawn($f,(hssA.A11, Qtree.left), true)
        hssA.A22 = $f(hssA.A22, Qtree.right)
        hssA.A11 = fetch(task)
        hssA.sz1 = size(hssA.A11); hssA.sz2 = size(hssA.A22)
        return hssA
      end
    end
  end
end

## extractor routines
_extract_nrows_kernel(hssA::HssMatrix, ntree::BinaryNode{Int}) = HssMatrix(hssA.D[1:ntree.data,:], hssA.U[1:ntree.data,:], hssA.V, hssA.rootnode)
_extract_crows_kernel(hssA::HssMatrix, ntree::BinaryNode{Int}) = HssMatrix(hssA.D[ntree.data+1:end,:], hssA.U[ntree.data+1:end,:], hssA.V, hssA.rootnode)
_extract_ncols_kernel(hssA::HssMatrix, ntree::BinaryNode{Int}) = HssMatrix(hssA.D[:,1:ntree.data], hssA.U, hssA.V[1:ntree.data,:], hssA.rootnode)
_extract_ccols_kernel(hssA::HssMatrix, ntree::BinaryNode{Int}) = HssMatrix(hssA.D[:,ntree.data+1:end], hssA.U, hssA.V[ntree.data+1:end,:], hssA.rootnode)

# generate branch routines via metaprogramming
for (f,g) in zip((:_extract_nrows, :_extract_crows, :_extract_ncols, :_extract_ccols), (:_extract_nrows_kernel, :_extract_crows_kernel, :_extract_ncols_kernel, :_extract_ccols_kernel)) 
  @eval begin
    function $f(hssA::HssMatrix, ntree::BinaryNode{Int})
      if isleaf(hssA)
        hssA = $g(hssA, ntree)
      else
        task = RecursionTools.spawn($f, (hssA.A11, ntree.left), true)
        A22 = $f(hssA.A22, ntree.right)
        A11 = RecursionTools.fetch(task)
        return HssMatrix(A11, A22, hssA.B12, hssA.B21, hssA.R1, hssA.W1, hssA.R2, hssA.W2, hssA.rootnode)
      end
    end
  end
end

## unpacking routine: unpacks hssB and adds it to hssA in place to contribute solution from lower levels
function _unpackadd_rows!(hssA::HssMatrix{T}, hssB::HssMatrix{T}, ktree::BinaryNode{Int}) where T
  if isbranch(hssA)
    if isbranch(hssB)
      hssA.B12 = blkdiagm(hssA.B12, hssB.B12)
      hssA.B21 = blkdiagm(hssA.B21, hssB.B21)
      
      hssA.R1 = blkdiagm(hssA.R1, hssB.R1)
      hssA.W1 = blkdiagm(hssA.W1, hssB.W1)
      hssA.R2 = blkdiagm(hssA.R2, hssB.R2)
      hssA.W2 = blkdiagm(hssA.W2, hssB.W2)

      hssA.A11 = _unpackadd_rows!(hssA.A11, hssB.A11, ktree.left)
      hssA.A22 = _unpackadd_rows!(hssA.A22, hssB.A22, ktree.right)
      hssA.sz1 == size(hssA.A11) || error("dimensions got screwed up!")
      hssA.sz2 == size(hssA.A22) || error("dimensions got screwed up!")
    else
      isbranch(ktree) || throw(ArgumentError("didn't expect ktree to be a leaf node"))
      k1 = ktree.left.data; k2 = ktree.right.data
      k1 + k2 == size(hssB,1) || throw(DimensionMismatch("first dimension of D does not match the expected k1+k2 rows"))
      m1, n1 = hssA.sz1
      r, w = gensize(hssB)
    
      # perform the unpacking on the leaves of hssA
      hssA.B12 = blkdiagm(hssA.B12, zeros(T, r, w), Matrix{T}(I, k1, k1))
      hssA.B21 = blkdiagm(hssA.B21, zeros(T, r, w), Matrix{T}(I, k2, k2))
    
      hssA.R1 = [blkdiagm(hssA.R1, Matrix{T}(I,r,r)); zeros(T, k1, size(hssA.R1,2)+r)]
      hssA.W1 = [blkdiagm(hssA.W1, Matrix{T}(I,w,w)); zeros(T, k2, size(hssA.W1,2)+w)]
      hssA.R2 = [blkdiagm(hssA.R2, Matrix{T}(I,r,r)); zeros(T, k2, size(hssA.R2,2)+r)]
      hssA.W2 = [blkdiagm(hssA.W2, Matrix{T}(I,w,w)); zeros(T, k1, size(hssA.W2,2)+w)]
    
      # perform the extend-add operation on the leaves
      hssA.A11 = _unpackadd_rows!(hssA.A11, hssB.D[1:k1,1:n1], hssB.U[1:k1,:], hssB.V[1:n1,:], hssB.D[k1+1:end,1:n1]) # probably should add @view
      hssA.A22 = _unpackadd_rows!(hssA.A22, hssB.D[k1+1:end,n1+1:end], hssB.U[k1+1:end,:], hssB.V[n1+1:end,:], hssB.D[1:k1,n1+1:end])
    
      hssA.sz1 == size(hssA.A11) || error("dimensions got screwed up!")
      hssA.sz2 == size(hssA.A22) || error("dimensions got screwed up!")
    end
    return hssA
  else
    if isbranch(hssB)
      error("Something went wrong. Expected hssB to also be a leaf node")
    else
      k, rk = size(hssB.U);
      k <= size(hssA,1) || throw(DimensionMismatch("expected hssB to have less rows than hssA."))
      hssA.U = [hssA.U zeros(size(hssA.U,1), rk)]
      hssA.U[end-k+1:end, end-rk+1:end] .= hssB.U
      hssA.V = [hssA.V hssB.V]
      hssA.D[end-k+1:end, :] .= hssA.D[end-k+1:end, :] .+ hssB.D
      return hssA
    end
  end
end

function _unpackadd_rows!(hssA::HssMatrix, D::Matrix, U::Matrix, V::Matrix, Brow::Matrix)
  if isleaf(hssA)
    k = size(D,1);
    size(D,1) == size(U,1) || throw(DimensionMismatch("first dimension of U does not match first dimension of D. Expected $(size(D,1)), got $(size(U,1))"))
    size(D,2) == size(V,1)|| throw(DimensionMismatch("first dimension of V does not match second dimension of D. Expected $(size(D,2)), got $(size(V,1))"))
    m, n = size(hssA.D)

    # perform the extend-add operation on the leaves
    hssA.D[end-k+1:end, :] .= hssA.D[end-k+1:end, :] .+ D
    hssA.U = [hssA.U zeros(size(hssA.U,1), size(U,2)+k)]
    hssA.U[end-k+1:end, end-size(U,2)-k+1:1:end] = [U Matrix(I,k,k)]
    hssA.V = [hssA.V V Brow']
    return hssA
  else
    error("Expected hssA to be a leaf node")
  end
end