"""
An iterator that can be used to specify sequences from 1 to n which omit a
single value.
For example, OneToNWithout(5, 3) defines an iterator that traverses
the values 1, 2, 4, 5.

from: https://github.com/skarppinen/cpf-diff-init.git
"""
struct OneToNWithout
  n::Int
  not::Int
  length::Int
  function OneToNWithout(n::Int, not::Int)
    @assert n > 0 "`n` must be > 0";
    @assert not >= 0 "`not` must be >= 0";
    @assert not <= n "`not` must be <= `n`";
    new(n, not, not > 0 ? n-1 : n);
  end
end

function Base.iterate(iter::OneToNWithout, state::Tuple{Int, Int} = (1, 0))
  element, count = state;
  count >= iter.length && return nothing;
  element != iter.not && return (element, (element + 1, count + 1));
  (element + 1, (element + 2, count + 1));
end

Base.length(iter::OneToNWithout) = iter.length;
Base.eltype(::Type{OneToNWithout}) = Int;

PDMats.PDMat(x::StaticMatrix) = PDMat(x, cholesky(x))

# Needed for rand!(MvNormal) with StaticArrays
# PDMats.unwhiten!(A::PDMat, v::MVector) = (v .= A.chol.L*v)
# PDMats.unwhiten!(A::PDMat, v::MVector) = (v .= transpose(UpperTriangular(A.chol.factors))*v)
PDMats.unwhiten!(A::PDMat, v::MVector) = (v .= unwhiten(A, v))
# PDMats.unwhiten(A::PDMat, v::StaticVector) = A.chol.L*v
PDMats.unwhiten(A::PDMat, v::StaticVector) = transpose(UpperTriangular(SMatrix(A.chol.factors)))*v

# Used to sample Static array
Base.rand(rng::AbstractRNG, d::MvNormal{<:Any, <:Any, SA}) where {SA <: StaticArray} = d.μ + unwhiten(d.Σ, randn(rng, SA))
#function Random.rand!(rng::AbstractRNG, d::MvNormal{<:Any,SA,MA}, v::MA) where {N,T,SA,MA <: MVector{N,T}}
function Random.rand!(rng::AbstractRNG, d::MvNormal{T,<:PDMat{T,<:StaticMatrix{N,N,T}},<:StaticVector{N,T}}, v::StaticVector{N,T}) where {N,T}
    # println(typeof(v))
    # println(N, T)
    v .= SVector(d.μ) + unwhiten(d.Σ, randn(rng, SVector{N,T}))
end

function Base.copy!(dst::PDMat{T, <:MMatrix}, src::PDMat{T, <:AbstractMatrix}) where {T}
    copy!(dst.mat, src.mat)
    copy!(dst.chol, src.chol)
    return dst
end

function Distributions.MvNormal(Σ::PDMat{T, SA}) where {N,T,SA<:StaticArray{Tuple{N,N},T}}
    return MvNormal(zero(similar_type(SA, Size(N))), Σ)
end

@inline function copy!(dst::MvNormal{T,<:PDMat{T,<:MMatrix{N,N,T}},<:MVector{N,T}}, src::MvNormal{T,<:PDMat{T,<:StaticMatrix{N,N,T}},<:StaticVector{N,T}}) where {N,T}
    dst.μ .= src.μ
    copy!(dst.Σ, src.Σ)
    return dst
end

# Non-allocating logpdf for MvNormal using Staticarrays
@inline function Distributions.logpdf(d::MvNormal{<:Any, <:Any, <:StaticVector{N}}, v::StaticVector{N}) where N
    # z = d.Σ.chol.L\(v - d.μ)
    z = transpose(UpperTriangular(d.Σ.chol.factors)) \ (v - d.μ)
    return -0.5*(dot(z,z) + N*log(2π) + logdet(d.Σ))
end

function PDMats.X_A_Xt(a::PDMat, x::StaticArray)
    # cf = a.chol.U
    cf = UpperTriangular(a.chol.factors)
    z = x*transpose(cf)
    return z * transpose(z)
end

@inline PDMats.X_A_Xt(a::PDMat, x::UniformScaling) = abs2(x.λ)*a

function X_invA_Xt(a::PDMat, x::StaticArray)
    # cf = a.chol.U
    cf = UpperTriangular(a.chol.factors)
    z = x / cf
    # z = rdiv!(copy(x), cf)
    return z * transpose(z)
end

function PDMats._addscal!(r::Matrix, a::Matrix, b::StaticMatrix, c::Real)
    if c == one(c)
        for i = 1:length(a)
            @inbounds r[i] = a[i] + b[i]
        end
    else
        for i = 1:length(a)
            @inbounds r[i] = a[i] + b[i] * c
        end
    end
    return r
end

"""
    expnormalize!(out,w)
    expnormalize!(w)
- `out .= exp.(w)/sum(exp,w)`. Does not modify `w`
- If called with only one argument, `w` is modified in place
"""
function expnormalize!(we,w)
    offset,maxind = findmax(w)
    @turbo @. w -= offset
    LoopVectorization.vmap!(exp,we,w)
    @turbo w += offset
    s    = sum_all_but(we,maxind) # s = ∑wₑ-1
    @turbo we *= 1/(s+1)
end

function expnormalize!(w)
    offset,maxind = findmax(w)
    @turbo @. w -= offset
    LoopVectorization.vmap!(exp,w,w)
    s = sum_all_but(w,maxind) # s = ∑wₑ-1
    @turbo @. w *= 1/(s+1)
end

function logsumexp!(logW,W)
    offset,maxind = findmax(logW)
    @turbo @. logW  -= offset
    LoopVectorization.vmap!(exp,W,logW)
    s = sum_all_but(W,maxind) # s = ∑wₑ-1
    @turbo @. W *= 1/(s+1)
    @turbo @. logW  -= log1p(s)
    return log1p(s) + offset
end

function sum_all_but(w,i)
    w[i] -= 1
    s = sum(w)
    w[i] += 1
    return s
end

symmetrize(P) = (P + transpose(P))/2

"""
Symmetrize a matrix in-place by seting it to (A + A')/2
"""
function symmetrize!(A)
  for c in 1:size(A, 2)
    for r in 1:size(A, 1)
      A[r,c] += A[c,r]
      A[r,c] *= 0.5
      A[c,r] = A[r,c]
    end
  end
end


function chol_from_qr(Q::StaticArrays.QR)
    d = sign.(diag(Q.R))
    return Cholesky(d.*Q.R, 'U', 0)
end

function choldowndate!(A::Cholesky{T, <:StaticMatrix{N,N,T}}, X::StaticMatrix{N,M,T}) where {N,M,T}
    for m in 1:M
        choldowndate!(A, view(X, :, m))
    end
end

function choldowndate!(A::Cholesky{T,<:StaticMatrix{N,N,T}}, x::StaticVector{N, T}) where {N, T}
    U = A.factors
    for k in 1:N
        r = sqrt(U[k,k]^2 - x[k]^2)
        c = r / U[k,k]
        s = x[k] ./ U[k,k]
        U[k,k] = r
        if k < N
            u = view(U, k, k+1:N)
            xx = view(x, k+1:N)
            u .-= s.*xx
            u ./= c
            xx .*= c
            xx .-= s.*u
        end
    end
    return A
end

function cholupdate!(A::Cholesky{T,<:StaticMatrix{N,N,T}}, x::StaticVector{N, T}) where {N, T}
    L = A.factors
    for k in 1:N
        r = sqrt(L[k,k]^2 + x[k]^2)
        c = r / L[k, k]
        s = x[k] / L[k, k]
        L[k, k] = r
        if k < N
            L[k,k+1:end] = (L[k, k+1:end] .+ s.*x[k+1:end])./c
            x[k+1:end] = c.*x[k+1:end] .- s.*L[k, k+1:end]
        end
    end
    return A
end

function Base.copy!(dst::Cholesky{T,<:StaticMatrix{<:Any,<:Any,T}},
                    src::Cholesky{T,<:AbstractMatrix{T}}) where {T}
    if src.uplo !== dst.uplo
        copy!(dst.factors, tranpose(src.factors))
    else
        copy!(dst.factors, src.factors)
    end
    return dst
end
