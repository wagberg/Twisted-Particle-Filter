struct MultinomialResampler <: Resampler end

Base.show(io::IO, ::MultinomialResampler) = print(io, "Multinomial resampler")

function resample!(::MultinomialResampler, A, W)
   resample_multinomial(A, W)
end

struct ConditionalMultinomialResampler <: ConditionalResampler end
Base.show(io::IO, ::ConditionalMultinomialResampler) = print(io, "Conditional Multinomial resampler")
conditional(::MultinomialResampler) = ConditionalMultinomialResampler()

function resample!(::ConditionalMultinomialResampler, A, W)
    N = length(A)
    @inbounds A[1] = 1
    resample_multinomial(view(A, 2:N), W)
end

# All implementations below are O(n) but using uniform spacings was fastest when testing
resample_multinomial(a,w) = resample_multinomial(Random.GLOBAL_RNG, a, w)
"""
  resample_multinomial(rng, a, w)

Generate a ∼ Categorical(w) in-place.

Using sorted uniforms via uniform spacing.

**Non-Uniform Random Variate Generation**, _Devroye, Luc_, (Springer-Verlag, 1986, page 214)
"""
function resample_multinomial(rng::AbstractRNG, a, w)
    N = length(a) # number of samples to generate
    q = randexp(rng, N+1)
    cumsum!(q,q)
    @turbo @. q /= q[end]
    i = 1
    @inbounds s = w[1]
    @inbounds for n ∈ 1:N
        while s < q[n]
            i += 1
            s += w[i]
        end
        a[n] = i
    end
    shuffle!(a) # Shuffle is not always necessary but never wrong.
    return nothing
end

"""
  resample_multinomial(rng, a, w)

Generate a ∼ Categorical(w) in-place.

Using sorting.

**Non-Uniform Random Variate Generation**, _Devroye, Luc_, (Springer-Verlag, 1986, page 214)
"""
# function resample_multinomial(rng::AbstractRNG, a, w)
#     N = length(a)
#     U = sort!(rand(rng, N))
#     i = 1
#     s = w[1]
#     for n ∈ 1:N
#         while s < U[n]
#             i += 1
#             s += w[i]
#         end
#         a[n] = i
#     end
#     shuffle!(a)
#     return nothing
# end

"""
  resample_multinomial(rng, a, w)

Generate a ∼ Categorical(w) in-place.

Using algorithm fro Distributions.
"""
# function resample_multinomial(rng::AbstractRNG, a, w)
#     rand!(rng, Distributions.sampler(Distributions.Categorical(w)), a)
#     return nothing
# end

"""
  resample_multinomial(rng, a, w)

Generate a ∼ Categorical(w) in-place.

Using sorted uniforms via exponential spacings.
Using the fact that min(U₁,U₂,...,Uₙ) ∼ Beta(1, n).

**Non-Uniform Random Variate Generation**, _Devroye, Luc_, (Springer-Verlag, 1986, page 214)
"""
# function resample_multinomial(rng::AbstractRNG, a, w)
#     N = length(a)
#     U_ = 1.0
#     n_ = N

#     i = 1
#     s = w[1]
#     for n ∈ 1:N
#         U_ *= rand(rng)^(1/n_)
#         n_ -= 1
#         U = 1.0 - U_
#         while s < U
#             i += 1
#             s += w[i]
#         end
#         a[n] = i
#     end
#     shuffle!(a)
#     return nothing
# end

"""
  resample_multinomial(rng, a, w)

Generate a ∼ Categorical(w) in-place.

Using sorted uniforms via exponential spacings.
Using the fact that min(U₁,U₂,...,Uₙ) ∼ Beta(1, n).

**Non-Uniform Random Variate Generation**, _Devroye, Luc_, (Springer-Verlag, 1986, page 214)
"""
# function resample_multinomial5(rng::AbstractRNG, a, w)
#     N = length(a)
#     logU_ = 0.0
#     n_ = N
#     i = 1
#     @inbounds s = w[1]
#     @inbounds for n ∈ 1:N
#         logU_ += log(rand(rng))/n_
#         n_ -= 1
#         U = 1.0 - exp(logU_)
#         while s < U
#             i += 1
#             s += w[i]
#         end
#         a[n] = i
#     end
#     shuffle!(a)
#     return nothing
# end
