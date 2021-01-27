### Compute the ULV factorization of a HSS matrix
#
# as seen in
# Chandrasekaran, S., Gu, M., & Pals, T. (2006). A Fast $ULV$ Decomposition Solver for Hierarchically Semiseparable Representations.
# SIAM Journal on Matrix Analysis and Applications, 28(3), 603–622. https://doi.org/10.1137/S0895479803436652
#
# Written by Boris Bonev, Nov. 2020

# load efficient BLAS and LAPACK routines for factorizations
import LinearAlgebra.LAPACK.geqlf!
import LinearAlgebra.LAPACK.gelqf!
import LinearAlgebra.LAPACK.ormql!
import LinearAlgebra.LAPACK.ormlq!
import LinearAlgebra.BLAS.trsm

# custom datastructure to store the hierarchical ULV factorization
# the question is whether this is efficient or not
mutable struct ULVFactor{T<:Number}
  # data to store the factorization
  L1::Matrix{T}
  L2::Matrix{T}
  U::Matrix{T}
  V::Matrix{T}
  QU::Tuple{Matrix{T},Vector{T}}
  QV::Tuple{Matrix{T},Vector{T}}
  ind::Vector{Int}
  cind::Vector{Int}
  oind::Vector{Int}

  # indicators to help the recursion
  rootnode::Bool
  leafnode::Bool

  # the treestructure itself
  parent::ULVFactor{T}
  left::ULVFactor{T}
  right::ULVFactor{T}

  # Root constructor
  ULVFactor{T}() where T = (x = new{T}(); x.leafnode = true; x.rootnode = false; return x)

  # Child node constructor
  ULVFactor(p::ULVFactor{T}) where T = (x = new{T}(); x.parent = p; x.leafnode = true; x.rootnode = false; return x)
  ULVFactor(cl::ULVFactor{T}, cr::ULVFactor{T}) where T = (x = new{T}(); x.left = cl; x.right = cr; x.leafnode = false; x.rootnode = false; return x)
end
#ULVFactor(L, U, V) = ULVFactor{typeof(data)}(data)

#function ulvfactor(hssA::HssMatrix{T}) where T
#end


# write routine for multiplication with ULVFactor


# function for direct solution using the implicit ULV factorization
function ulvfactsolve(hssA::HssMatrix{T}, b::Matrix{T}) where T
  if hssA.leafnode && hssA.rootnode # exit early if it also happens to be the rootnode
    x = hssA.D\b
  else
    x = zeros(size(hssA,2), size(b,2))
    _ = _ulvfactsolve!(hssA, copy(b), x, 0, 0)
  end
  return x
end

function _ulvfactsolve!(hssA::HssMatrix{T}, b::Matrix{T}, z::Matrix{T}, ro::Int, co::Int) where T
  T <: Complex ? adj = 'C' : adj = 'T'
  if hssA.leafnode
    # create the factorization node in the factorization tree
    ulvA = ULVFactor{T}()
    # determine dimensions
    m, n = size(hssA.D)
    k = size(hssA.U, 2)
    nk = min(m-k,n)
    ind = 1:m-k
    cind = m-k+1:m
    cols = co .+ (1:n)
    # can't be compressed, exit early
    if k >= m
      u = zeros(m, size(b,2))
      D = hssA.D; U = hssA.U; V = hssA.V
    else
      # form QL decomposition of the row generators and apply it
      U = copy(hssA.U)
      qlf = geqlf!(U);
      U = tril(U[end-k+1:end,:]) # k x k block
      D = ormql!('L', adj, qlf..., copy(hssA.D)) # transform the diagonal block
      ormql!('L', adj, qlf..., b) # transform the right-hand side
      # Form the LQ decomposition of the first m-k rows of D
      lqf = gelqf!(D[1:end-k,:])
      L1 = tril(lqf[1]); L1 = L1[:,1:nk]
      L2 = ormlq!('R', adj, lqf..., copy(D[end-k+1:end,:])) # update the bottom block of the diagonal block # TODO: remove the copy as it's prob. unnecessary
      zloc = trsm('L', 'L', 'N', 'N', 1., L1, b[ind,:])
      b = b[cind, :] .- L2[:,1:nk] * zloc # remove contribution in the uncompressed parts
      V = ormlq!('L', 'N', lqf..., copy(hssA.V)) # compute the updated off-diagonal generators on the right
      u = V[ind,:]' * zloc # compute update vector to be passed on
      # pass on uncompressed parts of the problem
      D = L2[:,nk+1:end]
      V = V[nk+1:end,:]
      z[cols[ind], :] = zloc
      ulvA.QV = lqf; ulvA.oind = cols
    end
  else
    b1 = b[1:hssA.m1, :]; b2 = b[hssA.m1+1:end, :]
    b1, u1, D1, U1, V1, cols1, nk1, ulvA1 = _ulvfactsolve!(hssA.A11, b1, z, ro, co)
    b2, u2, D2, U2, V2, cols2, nk2, ulvA2 = _ulvfactsolve!(hssA.A22, b2, z, ro+hssA.m1, co+hssA.n1)
    ulvA = ULVFactor(ulvA1, ulvA2)

    # merge nodes to form new diagonal block 
    b = [b1; b2] .- [U1*hssA.B12*u2; U2*hssA.B21*u1]
    D = [D1 U1*hssA.B12*V2'; U2*hssA.B21*V1' D2]
    m, n = size(D)
    m1 = hssA.m1; n1 = hssA.n1; m2 = hssA.m2; n2 = hssA.n2
    cols = [cols1[nk1+1:end]; cols2[nk2+1:end]] # to re-adjust local numbering

    # early exit if topnode
    if !hssA.rootnode
      # if not the topnode we continue merging off-diagonal blocks and compressing them
      U = [U1*hssA.R1; U2*hssA.R2]
      V = [V1*hssA.W1; V2*hssA.W2]
      k = size(U,2)
      nk = min(m-k,n)
      ind = 1:m-k
      cind = m-k+1:m
      # can't be compressed, exit early
      if k >= m
        u = hssA.W1'*u1 .+ hssA.W2'*u2
      else
        # form QL decomposition of the row generators
        qlf = geqlf!(U);
        U = tril(U[end-k+1:end,:]) # k x k block
        # transform the diagonal block
        ormql!('L', adj, qlf..., D)
        ormql!('L', adj, qlf..., b) # transform the right-hand side
        # Form the LQ decomposition of the first m-k rows of D
        lqf = gelqf!(D[1:end-k,:])
        L1 = tril(lqf[1]); L1 = L1[:,1:nk]
        L2 = ormlq!('R', adj, lqf..., copy(D[end-k+1:end,:])) # update the bottom block of the diagonal block
        zloc = trsm('L', 'L', 'N', 'N', 1., L1, b[ind,:])
        b = b[cind, :] .- L2[:,1:nk] * zloc # adjust right-hand side
        V = ormlq!('L', 'N', lqf..., V)
        # pass on uncompressed parts of the problem
        u = V[ind,:]' * zloc .+ hssA.W1'*u1 .+ hssA.W2'*u2
        D = L2[:, nk+1:end]
        V = V[nk+1:end,:]
        # Prepare things to pass on
        z[cols[ind], :] = zloc
        ulvA.QV = lqf; ulvA.oind = cols
      end
    else
      z[cols, :] = D\b
      _ulvsolve_topdown!(ulvA, z)
      u = 0; U = 0; V = 0; nk = 0
    end
  end
  return b, u, D, U, V, cols, nk, ulvA
end

function _ulvsolve_topdown!(ulvA::ULVFactor{T}, z::Matrix{T}) where T
  T <: Complex ? adj = 'C' : adj = 'T'
  if isdefined(ulvA, :QV)
    z[ulvA.oind,:] = ormlq!('L', adj, ulvA.QV..., z[ulvA.oind,:])
  end
  if !ulvA.leafnode
    _ulvsolve_topdown!(ulvA.left, z)
    _ulvsolve_topdown!(ulvA.right, z)
  end
end


# temporary name for function that actually just computes the ULV factorization
# function hssdivide(hssA::HssMatrix{T}, hssB::HssMatrix{T}) where T
# end

# function _hssdivide(hssA::HssMatrix{T})
# end