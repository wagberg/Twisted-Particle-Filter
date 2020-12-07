## Multinomial resampling.
# The implementation is based on some code by Matti Vihola.
"""
An abstract type representing a resampling algorithm.
"""
abstract type Resampling end
struct MultinomialResampling <: Resampling end
struct SystematicResampling <: Resampling end

function resample!(ind::AVec{Int}, w::AVec{<:AFloat}, ::SystematicResampling, conditional=false)::Nothing
  if conditional
    N = length(ind)
    @inbounds q = N*w[1]
    if @inbounds q <= 1
       U = q*rand()
    else
      r = mod(q, 1)
      U = r*ceil(q)/q < rand() ? r*rand() : r + (1-r)*rand()
    end
  else
    U = rand()
  end
  _systematic_resample!(ind, w, U)
end

function _systematic_resample!(ind::AVec{Int}, w::AVec{<:AFloat}, U::Float64)::Nothing
  N = length(ind)
  v = 0.0
  m = 0
  for n in 1:N
    while v < U
      m += 1
      @inbounds v += N*w[m]
    end
    @inbounds ind[n] = m
    U += 1.0
  end
end


function resample!(ind::AVec{Int}, w::AVec{<:AFloat}, ::MultinomialResampling, conditional::Bool=false)::Nothing
  if conditional
    @inbounds ind[1] = 1
    @views _multinomial_resample(ind[2:end], w)
  else
    _multinomial_resample(ind, w)
  end
  nothing
end

function _multinomial_resample(ind, w)
  N = length(ind)
  q = cumsum(randexp(N+1))
  q./ q[end]
  @inbounds s = w[1]
  i = one(eltype(ind))
  for n in 1:N
    @inbounds while s < q[n]
      i += 1
      @inbounds s = w[i]
    end
    @inbounds ind[n] = i
  end
  shuffle!(ind)
  nothing
end



# function resample!(ind::AbstractVector{Int64}, p::AbstractVector{<: Real}, ::MultinomialResampling;
#                    ref_prev::Int = 0, ref_cur::Int = 0)::Nothing
#     m = length(p);
#     n = length(ind);
#     K = 1; @inbounds S = p[1];
#     U_ = 1.0; n_ = n;
#     for j in 1:n
#       # Order statistics in reverse order:
#       U_ = rand() ^ (1.0 / n_) * U_;
#       n_ -= 1;
#       # ...same as forward order by symmetry...
#       U = 1.0 - U_;
#       # Find K such that F(K) >= u
#       while K < m && U > S
#         K = K + 1;       # Note that K is not reset!
#         @inbounds S = S + p[K]; # S is the partial sum up to K
#       end
#       @inbounds ind[j] = K;
#     end

#     if ref_prev > 0
#       # Need swap or shuffle because of sampling uniforms in order.
#       #swap!(ind, ref_cur, sample(Base.OneTo(m)));
#       shuffle!(ind);
#       @inbounds ind[ref_cur] = ref_prev;
#     end
#     nothing
# end

# function swap!(x::AbstractVector, i::Integer, j::Integer)
#   @inbounds tmp = x[i];
#   @inbounds x[i] = x[j];
#   @inbounds x[j] = tmp;
#   nothing
# end