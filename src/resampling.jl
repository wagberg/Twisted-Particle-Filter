## Multinomial resampling.
# The implementation is based on some code by Matti Vihola.
"""
An abstract type representing a resampling algorithm.
"""
abstract type Resampling end
struct MultinomialResampling <: Resampling end

function resample!(ind::AbstractVector{Int64}, p::AbstractVector{<: Real}, mr::MultinomialResampling;
                   ref_prev::Int = 0, ref_cur::Int = 0)::Nothing
    m = length(p);
    n = length(ind);
    K = 1; @inbounds S = p[1];
    U_ = 1.0; n_ = n;
    for j in 1:n
      # Order statistics in reverse order:
      U_ = rand() ^ (1.0 / n_) * U_;
      n_ -= 1;
      # ...same as forward order by symmetry...
      U = 1.0 - U_;
      # Find K such that F(K) >= u
      while K < m && U > S
        K = K + 1;       # Note that K is not reset!
        @inbounds S = S + p[K]; # S is the partial sum up to K
      end
      @inbounds ind[j] = K;
    end

    if ref_prev > 0
      # Need swap or shuffle because of sampling uniforms in order.
      #swap!(ind, ref_cur, sample(Base.OneTo(m)));
      shuffle!(ind);
      @inbounds ind[ref_cur] = ref_prev;
    end
    nothing
end

function swap!(x::AbstractVector, i::Integer, j::Integer)
  @inbounds tmp = x[i];
  @inbounds x[i] = x[j];
  @inbounds x[j] = tmp;
  nothing
end


"""
    logΣexp, Σ = logsumexp!(p::WeightedParticles)
Return log(∑exp(w)). Modifies the weight vector to `w = exp(w-offset)`
Uses a numerically stable algorithm with offset to control for overflow and `log1p` to control for underflow.
References:
https://arxiv.org/pdf/1412.8695.pdf eq 3.8 for p(y)
https://discourse.julialang.org/t/fast-logsumexp/22827/7?u=baggepinnen for stable logsumexp
"""
function logsumexp!(w)
    N = length(w)
    offset, maxind = findmax(w)
    w .= exp.(w .- offset)
    Σ = sum_all_but(w,maxind) # Σ = ∑wₑ-1
    log1p(Σ) + offset, Σ+1
end

function sum_all_but(w,i)
    w[i] -= 1
    s = sum(w)
    w[i] += 1
    s
end

"""
    loglik = resample!(p::WeightedParticles)
Resample the particles based on the `p.logweights`. After a call to this function, weights will be reset to sum to one. Returns log-likelihood.
"""
function resample!(w, ξ)
    N = length(w)
    logΣexp,Σ = logsumexp!(w)
    _resample!(ξ,w,Σ)
    fill!(w, -log(N))
    logΣexp - log(N)
end



"""
In-place systematic resampling of `p`, returns the sum of weights.
`p.logweights` should be exponentiated before calling this function.
"""
function _resample!(ξ,w,Σ)
    N = length(w)
    bin = w[1]
    s = rand()*Σ/N
    bo = 1
    for i = 1:N
        @inbounds for b = bo:N
            if s < bin
                ξ[i] = ξ[b]
                bo = b
                break
            end
            bin += w[b+1] # should never reach here when b==N
        end
        s += Σ/N
    end
    Σ
end

function resample_multinomial(w::AbstractVector{<:Real}, num_particles::Integer)
    return rand(Distributions.sampler(Categorical(w)), num_particles)
end