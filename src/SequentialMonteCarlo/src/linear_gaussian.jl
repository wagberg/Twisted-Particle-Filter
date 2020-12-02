"""
A linear Gaussian state-space model parameterized by the state dimension
"""
struct LGSSM{N} <: SSM{FloatParticle{N}}
end

function default_parameters(::LGSSM{N}) where N
    LGSSMParameter(N, 1, 1)()
end

@with_kw struct LGSSMParameter{nx,nu,ny}
    A::SMatrix{nx,nx} = SMatrix{nx,nx}(Diagonal(rand(nx)))
    B::SMatrix{nx,nu} = SMatrix{nx,nu}(rand(nx * nu))
    C::SMatrix{ny,nx} = SMatrix{ny,nx}(rand(ny * nx))
    D::SMatrix{ny,nu} = SMatrix{ny,nu}(zeros(ny * nu))
    Q::Symmetric      = Symmetric(Diagonal(rand(nx)))
    R::Symmetric      = Symmetric(Diagonal(rand(ny)))
    μ0::SVector{nx}   = SVector{nx}(randn(nx))
    Σ0::Symmetric     = Symmetric(Diagonal(rand(nx)))
end

function transition_function(x, ::LGSSM, t::Integer, data, θ)
    @unpack A, B = θ
    A * x .+ B * data.u[t]
end

function observation_function(x, ::LGSSM, t::Integer, data, θ)
    @unpack C, D = θ
    C * x .+ D * data.u[t]
end

function transition_state_jacobian(x, model::LGSSM, t::Integer, data, θ)
    @unpack A = θ
    A
end

function observation_state_jacobian(x, model::LGSSM, t::Integer, data, θ)
    @unpack C = θ
    C
end