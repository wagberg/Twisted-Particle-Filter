### Motion/Measurement Models ###

"""
`DynamicsModel` is an abstract type to encapsulate linear
and nonlinear dynamics models
"""
abstract type DynamicsModel end

"""
`ObservationModel` is an abstract type to encapsulate linear
and nonlinear observation models
"""
abstract type ObservationModel end

abstract type AbstractJacobian end
struct NoiseJacobian <: AbstractJacobian end
struct StateJacobian <: AbstractJacobian end


predict(m::DynamicsModel, x::AbstractVector{<:Number}; u::AbstractVector{<:Number} = zeros(eltype(x), 0)) = predict(m, x, u, zeros(eltype(x),size(m.d)))
predict(m::DynamicsModel, x::AbstractVector, rng::AbstractRNG; u::AbstractVector = zeros(eltype(x), 0)) = predict(m, x, u, rand(rng, m.d))

jacobian(::NoiseJacobian, m::DynamicsModel, x; u=zeros(eltype(x),0)) = ForwardDiff.jacobian(w->predict(m, x, u, w), zeros(eltype(m.d),size(m.d)))
jacobian(::StateJacobian, m::DynamicsModel, x; u=zeros(eltype(x),0)) = ForwardDiff.jacobian(x->predict(m, x, u=u), x)
jacobian(m::DynamicsModel, x; u=zeros(eltype(x), 0)) = jacobian(StateJacobian(), m, x, u=u)

measure(m::ObservationModel, x::AbstractVector{<:Number};  u::AbstractVector{<:Number} = zeros(eltype(x), 0)) = measure(m, x, u, zeros(eltype(m.d), size(m.d)))
measure(m::ObservationModel, x::AbstractVector{<:Number}, rng::AbstractRNG; u::AbstractVector{<:Number}=zeros(eltype(x), 0)) = measure(m, x, u, rand(rng, m.d))

jacobian(::NoiseJacobian, m::ObservationModel, x; u=zeros(eltype(x),0)) = ForwardDiff.jacobian(w->measure(m, x, u, w), zeros(eltype(m.d),size(m.d)))
jacobian(::StateJacobian, m::ObservationModel, x; u=zeros(eltype(x),0)) = ForwardDiff.jacobian(x->measure(m, x, u=u), x)
jacobian(m::ObservationModel, x; u=zeros(eltype(x), 0)) = jacobian(StateJacobian(), m, x, u=u)


"""
An abstract type for an MVector or SVector with length N and element type T.
"""
const VecND{N, T} = StaticArray{Tuple{N}, T, 1};

"""
An abstract type for an N * M MMatrix or SMatrix with element type T.
"""
const MatNMD{N, M, T} = StaticArray{Tuple{N, M}, T, 2};

"""
An abstract type for an N * N MMatrix or SMatrix with element type T.
"""
const MatND{N, T} = StaticArray{Tuple{N, N}, T, 2};


"""
    LinearDynamicsModel(A::AbstractMatrix,B::AbstractMatrix,W::Symmetric)
    LinearDynamicsModel(A::AbstractMatrix,B::AbstractMatrix,W::AbstractMatrix)

Construct linear dynamics model with; transition matrix A,
control matrix B, and symmetric zero-mean process noise with
symmetric covariance matrix W
"""
struct LinearDynamicsModel{TA, TB, TM, Td<:MvNormal} <: DynamicsModel
    A::TA
    B::TB
    M::TM
    d::Td
end

function LinearDynamicsModel(A::AbstractMatrix{<:Number}, B::AbstractMatrix{<:Number},
    M::AbstractMatrix{<:Number}, Q::AbstractMatrix{<:Number})
    @assert size(A,1) == size(A,2) == size(B,1) == size(M,1) == size(Q,1) == size(Q,2) "Dimension mismatch"
    n = size(A,1)
    m = size(B,2)
    d = size(M,2)
    LinearDynamicsModel(SMatrix{n,n}(A),SMatrix{n,m}(B),SMatrix{n,d}(M),MvNormal(Symmetric(Q)))
end

LinearDynamicsModel(A::AbstractMatrix{<:Number}, B::AbstractMatrix{<:Number}, Q::AbstractMatrix{<:Number}) = LinearDynamicsModel(A, B, typeof(A)(I, size(Q)...), Q)
LinearDynamicsModel(A::AbstractMatrix{<:Number}, Q::AbstractMatrix{<:Number}) = LinearDynamicsModel(A, zeros(eltype(A), size(A,1), 0), typeof(A)(I, size(Q)...), Q)

"""
    predict(m::LinearDynamicsModel, x::AbstractVector{<:Number}, u::AbstractVector{<:Number})
    predict(m::LinearDynamicsModel, x::AbstractVector{<:Number}, u::AbstractVector{<:Number}, rng::AbstractRNG)

Uses the linear dynamics model to propagate the state x one step forward in time with control input u.
If rng is given, it adds process noise. 
"""
function predict(m::LinearDynamicsModel, x::AbstractVector{<:Number}, 
                 u::AbstractVector{<:Number}, w::AbstractVector{<:Number})
    return m.A * x .+ m.B * u .+ m.M * w
end

jacobian(::StateJacobian, m::LinearDynamicsModel, x; u=zeros(eltype(x),0)) = m.A
jacobian(::NoiseJacobian, m::LinearDynamicsModel, x; u=zeros(eltype(x),0)) = m.M

"""
    LinearObservationModel(C::AbstractMatrix,D::AbstractMatrix,V::Symmetric)
    LinearObservationModel(C::AbstractMatrix,D::AbstractMatrix,V::AbstractMatrix)
    LinearObservationModel(C::AbstractMatrix,V::Symmetric)
    LinearObservationModel(C::AbstractMatrix,V::AbstractMatrix)

Construct linear observation dynamics model with; transition matrix C,
control matrix B, and symmetric zero-mean measurement noise with
symmetric covariance matrix V
"""
struct LinearObservationModel{TC,TD, TN, S<:MvNormal} <: ObservationModel
    C::TC
    D::TD
    N::TN
    d::S    
end

function LinearObservationModel(C, D, N, R)
    @assert size(C,1) == size(D,1) == size(N,1) "Dimension mismatch"
    @assert size(N,2) == size(R,1) == size(R,2) "Dimension mismatch"
    ny, nx = size(C)
    nu = size(D,2)
    nv = size(N,2)
    LinearObservationModel(SMatrix{ny,nx}(C),SMatrix{ny,nu}(D),SMatrix{ny,nv}(N),MvNormal(Symmetric(R)))
end

LinearObservationModel(C, D, R) = LinearObservationModel(C, D, typeof(C)(I, size(R)...), R)
LinearObservationModel(C, R) = LinearObservationModel(C, zeros(eltype(C), size(C,1), 0), R)

"""
    measure(m::LinearObservationModel, x::AbstractVector{<:Number}, u::AbstractVector{<:Number})
    measure(m::LinearObservationModel, x::AbstractVector{T}, u::AbstractVector{T}, rng::AbstractRNG) where T<:Number

Returns an observation of state x according to the linear observation model m, with control inputs u.
If rng is passed, adds additive Gaussian noise to the observation.
"""
function measure(m::LinearObservationModel, x::AbstractVector{<:Number}, 
                 u::AbstractVector{<:Number}, w::AbstractVector{<:Number})
    return m.C * x .+ m.D * u .+ m.N * w
end

jacobian(::StateJacobian, m::LinearObservationModel, x; u=zeros(eltype(x),0)) = m.C
jacobian(::NoiseJacobian, m::LinearObservationModel, x; u=zeros(eltype(x),0)) = m.N

"""
    NonlinearDynamicsModel(f::Function,d::Distribution)

Construct nonlinear dynamics model with transition function f
and symmetric zero-mean process noise with symmetric covariance matrix W
"""
struct NonlinearDynamicsModel{T<:Function, S<:Distribution} <: DynamicsModel
    f::T
    d::S
end

"""
    predict(m::NonLinearDynamicsModel, x::AbstractVector{<:Number}, u::AbstractVector{<:Number})
    predict(m::NonLinearDynamicsModel, x::AbstractVector{<:Number}, u::AbstractVector{<:Number}, rng::AbstractRNG)

Uses the non linear dynamics model to propagate the state x one step forward in time with control input u.
If rng is given, it adds process noise. 
"""
function predict(m::NonlinearDynamicsModel, x::AbstractVector{<:Number}, 
                 u::AbstractVector{<:Number}, w::AbstractVector{<:Number})
    return m.f(x, u, w)
end

"""
    NonlinearObservationModel(h::Function,V::Symmetric)
    NonlinearObservationModel(h::Function,V::AbstractMatrix)

Construct nonlinear observation dynamics model with measurement function h
and symmetric zero-mean measurement noise with symmetric covariance matrix V
"""
struct NonlinearObservationModel{T<:Function, S<:Distribution} <: ObservationModel
    h::T
    d::S
end

# NonlinearObservationModel(h::Function, R::Symmetric) = NonlinearObservationModel(h, MvNormal(R))
# NonlinearObservationModel(h::Function, R::AbstractMatrix) = NonlinearObservationModel(h, Symmetric(R))

"""
    measure(m::LinearObservationModel, x::AbstractVector{<:Number}, u::AbstractVector{<:Number})
    measure(m::LinearObservationModel, x::AbstractVector{T}, u::AbstractVector{T}, rng::AbstractRNG) where T<:Number

Returns an observation of state x according to the non linear observation model m, with control inputs u.
If rng is passed, adds additive Gaussian noise to the observation.
"""
function measure(m::NonlinearObservationModel, x::AbstractVector{<:Number},
    u::AbstractVector{<:Number}, e::AbstractVector{<:Number})
    return m.h(x, u, e)
end