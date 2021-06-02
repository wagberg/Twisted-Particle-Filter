abstract type Particle end

"""
Returns the names of the states.
"""
statenames(::Type{P}) where P <: Particle = fieldnames(P);
"""
Copy particle mutating dst.
"""
function copy!(dst::Particle, src::Particle) end
"""
Swap contents of particles.
"""
function swap!(dst::Particle, src::Particle) end
"""
Convert particle to a static vecotr.
"""
function toSvector(x::Particle) end

function ==(p1::P, p2::P) where P <: Particle
    toSVector(p1) == toSVector(p2)
end

"""
Real valued particle of dimension __N__.
* `N` - state dimension
"""
struct FloatParticle{N,T}  <: Particle
    x::MVector{N,T}
end

"""
Initialize all states to zero.
"""
FloatParticle{N, T}() where {N, T} = FloatParticle(zero(MVector{N, T}))
FloatParticle{N}() where {N} = FloatParticle{N, Float64}()

function copy!(dst::FloatParticle{N, T}, src::FloatParticle{N, T}) where {N, T}
    copy!(dst.x, src.x)
    return dst
end
copy!(dst::FloatParticle, src::StaticArray) = copy!(dst.x, src)

"""
Swap particles.
"""
function swap!(dst::FloatParticle{N, T}, src::FloatParticle{N, T}) where {N, T}
    for i in eachindex(src.x)
        src.x[i], dst.x[i] = dst.x[i], src.x[i]
    end
end

"""
Convert the particle to a static vector.
This is usefull in analysis and plotting.
"""
function toSVector(p::FloatParticle{N, T}) where {N, T}
    SVector{N, T}(p.x)
end

function statenames(::Type{FloatParticle{N, T}}) where {N, T}
    Symbol.("x" .* string.(collect(1:N)))
end
