struct SystematicResampler <: Resampler end

Base.show(io::IO, ::SystematicResampler) = print(io, "Systematic resampling")

function resample!(::SystematicResampler, A, W)
    resample_systematic(A, W, rand())
    return nothing
end

struct ConditionalSystematicResampler <: ConditionalResampler end
conditional(r::SystematicResampler) = ConditionalSystematicResampler()

function resample!(::ConditionalSystematicResampler, a, w)
    N = length(a)
    q = N*w[1]
    if q <= 1.0
        U = q*rand()
    else
        r = mod(q, 1)
        U = r*ceil(q)/q < rand() ? r*rand() : r + (1.0-r)*rand()
    end
    return resample_systematic(a, w, U)
end
"""
Systematic resampler.

Samples ancestor indicies in A given weights in W.
"""
function resample_systematic(a, w, U)
    N = length(a)
    M = length(w)
    i = 1
    s = N*w[1]
    for n ∈ 1:N
        while s<U && i<M
            i += 1
            s += N*w[i]
        end
        a[n] = i
        U += 1.0
    end
    return nothing
end
