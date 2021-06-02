using Test, SequentialMonteCarlo, Distributions, Random, PDMats
using StaticArrays

function get_1D_kalman()
    # 1-dimensional example
    T = 100
    A = 0.5
    B = 0.3
    Q = 0.7
    C = 1.0
    D = 0.8
    R = 0.6
    μ = 0.0
    Σ = Q/(1-A^2)


    u = randn(T)
    y = zeros(T)
    x = μ + sqrt(Σ)*randn()
    y[1] = C*x + D*u[1] + sqrt(R)*randn()
    for t in 2:T
        x = A*x + B*u[t] + sqrt(Q)*randn()
        y[t] = C*x + D*u[t] + sqrt(R)*randn()
    end

    # Initialize
    mf = zeros(T)
    σf = zeros(T)
    mp = zeros(T+1)
    σp = zeros(T+1)
    mp[1] = μ
    σp[1] = Σ

    llc = 0.0
    ll = zeros(T)
    for t in 1:T
        # Correct
        v = y[t] - C*mp[t] - D*u[t]
        S = C^2*σp[t] + R
        ll[t] = llc += -0.5*log(2π*S) - 0.5*v^2/S
        K = σp[t]*C/S
        mf[t] = mp[t] + K*v
        σf[t] = σp[t] - K^2*S

        # Predict
        mp[t+1] = A*mf[t] + B*u[t]
        σp[t+1] = A^2*σf[t] + Q
    end
    return (;y,u,A,B,C,D,Q,R,μ,Σ,mf,σf,mp,σp,ll)
end

function get_1D_rts()
    y,u,A,B,C,D,Q,R,μ,Σ,mf,σf,mp,σp,ll = get_1D_kalman()

    T = length(y)
    ms = zeros(T)
    σs = zeros(T)

    ms[T] = mf[T]
    σs[T] = σf[T]
    for t in T-1:-1:1
        G = σf[t]*A/σp[t+1]
        ms[t] = mf[t] + G*(ms[t+1] - mp[t+1])
        σs[t] = σf[t] + G^2*(σs[t+1] - σp[t+1])
    end
    return (;y,u,A,B,C,D,Q,R,μ,Σ,mf,σf,mp,σp,ms,σs,ll)
end

function get_2D_Kalman()
    T = 100
    nx = 2
    ny = 1
    nu = 1
    A = [0.5 1.0; 0.0 0.6]
    B = reshape([0.3 0.6], :, 1)
    QL = [0.7  0.0; 0.3 0.6]
    Q =  QL*QL'
    C = reshape([1.0 -2.4], 1, :)
    D = reshape([0.8], 1, 1)
    RL = reshape([1.1], 1, 1)
    R = RL*RL'
    μ = [0.0, 0.0]
    ΣL = [1.0 0.0; 0.2 2.1]
    Σ = ΣL*ΣL'


    u = randn(nu, T)
    y = zeros(ny, T)
    x = μ .+ ΣL*randn(nx)
    y[:,1] .= C*x + D*u[:,1] + RL*randn(ny)
    for t in 2:T
        x = A*x + B*u[:,t] + QL*randn(nx)
        y[:,t] = C*x + D*u[:,t] + RL*randn(ny)
    end

    # Initialize
    mf = zeros(nx,T)
    Σf = zeros(nx,nx,T)
    mp = zeros(nx,T+1)
    Σp = zeros(nx,nx,T+1)
    mp[:,1] .= μ
    Σp[:,:,1] = Σ

    llc = 0.0
    ll = zeros(T)
    for t in 1:T
        # Correct
        # v = y[:,t] .- C*m .- D*u[:,t]
        v = y[:,t] .- C*mp[:,t] .- D*u[:,t]
        # S = C*P*C' .+ R
        S = C*Σp[:,:,t]*C' .+ R
        ll[t] = llc += logpdf(MvNormal(S), v)
        # ll[t] = llc += -ny/2*log(2π) - 1/2*log(det(S)) - 1/2*v'*(S\v)
        # K = P*C'/S
        K = Σp[:,:,t]*C'/S
        # m .= m .+ K*v
        mf[:,t] = mp[:,t] .+ K*v
        # P .-= K*S*K'
        Σf[:,:,t] = Σp[:,:,t].- K*S*K' |> x-> (x.+x')./2

        # Predict
        # m .= A*m
        mp[:,t+1] .= A*mf[:,t] .+ B*u[:,t]
        # P .= A*P*A' .+ Q
        Σp[:,:,t+1] .= A*Σf[:,:,t]*A' .+ Q |> x-> (x.+x')./2
    end
    return (;y,u,A,B,C,D,Q,R,μ,Σ,mf,Σf,mp,Σp,ll)
end

function get_2D_rts()
    y,u,A,B,C,D,Q,R,μ,Σ,mf,Σf,mp,Σp,ll = get_2D_Kalman()

    T = length(y)
    ms = zeros(2,T)
    Σs = zeros(2,2,T)

    ms[:,T] .= mf[:,T]
    Σs[:,:,T] .= Σf[:,:,T]

    for t in T-1:-1:1
        G = Σf[:,:,t]*A'/Σp[:,:,t+1]
        ms[:,t] .= mf[:,t] .+ G*(ms[:,t+1] .- mp[:,t+1])
        Σs[:,:,t] .= Σf[:,:,t] .+ G*(Σs[:,:,t+1] .- Σp[:,:,t+1])*G'
    end
    return (;y,u,A,B,C,D,Q,R,μ,Σ,mf,Σf,mp,Σp,ms,Σs,ll)
end

Random.seed!(42)
@testset "Kalman filter" begin

    @testset "1-D example" begin

        y,u,A,B,C,D,Q,R,μ,Σ,mf,σf,mp,σp,ll = get_1D_kalman()
        model = LinearGaussian{1,1,1}()
        θ = LinearGaussianPar(
            A=SMatrix{1,1}(A),
            B=SMatrix{1,1}(B),
            C=SMatrix{1,1}(C),
            D=SMatrix{1,1}(D),
            Q=PDMat(SMatrix{1,1}(Q)),
            R=PDMat(SMatrix{1,1}(R)),
            μ₀=SVector{1}(μ),
            Σ₀=PDMat(SMatrix{1,1}(Σ)))
        data = (;y=SVector.(y), u=SVector.(u))
        kf = KalmanFilter(model, data)
        kf(θ)
        @test log_likelihood(kf) ≈ ll
        @test mf ≈ reinterpret(Float64, SVector.(mean.(filter_density(kf))))
        @test mp ≈ reinterpret(Float64, SVector.(mean.(predictive_density(kf))))
        @test σf ≈ reinterpret(Float64, SVector.(var.(filter_density(kf))))
        @test σp ≈ reinterpret(Float64, SVector.(var.(predictive_density(kf))))
    end

    @testset "2-D example" begin
        # 1-dimensional example

        y,u,A,B,C,D,Q,R,μ,Σ,mf,Σf,mp,Σp,ll = get_2D_Kalman()

        model = LinearGaussian{2,1,1}()
        θ = LinearGaussianPar(
            A=SMatrix{2,2}(A),
            B=SMatrix{2,1}(B),
            C=SMatrix{1,2}(C),
            D=SMatrix{1,1}(D),
            Q=PDMat(SMatrix{2,2}(Q)),
            R=PDMat(SMatrix{1,1}(R)),
            μ₀=SVector{2}(μ),
            Σ₀=PDMat(SMatrix{2,2}(Σ)))
        data = (;y=SVector.(y), u=SVector.(u))
        kf = KalmanFilter(model, data)
        kf(θ)
        @test log_likelihood(kf) ≈ ll
        @test reinterpret(reshape, Float64, SVector.(mean.(predictive_density(kf)))) ≈ mp
        @test reinterpret(Float64, SMatrix.(cov.(predictive_density(kf)))) ≈ Σp[:]
        @test reinterpret(reshape, Float64, SVector.(mean.(filter_density(kf)))) ≈ mf
        @test reinterpret(Float64, SMatrix.(cov.(filter_density(kf)))) ≈ Σf[:]
    end
end

@testset "RTS Smoother" begin

    @testset "1D-example" begin
        y,u,A,B,C,D,Q,R,μ,Σ,mf,σf,mp,σp,ms,σs,ll = get_1D_rts()
        model = LinearGaussian{1,1,1}()
        θ = LinearGaussianPar(
            A=SMatrix{1,1}(A),
            B=SMatrix{1,1}(B),
            C=SMatrix{1,1}(C),
            D=SMatrix{1,1}(D),
            Q=PDMat(SMatrix{1,1}(Q)),
            R=PDMat(SMatrix{1,1}(R)),
            μ₀=SVector{1}(μ),
            Σ₀=PDMat(SMatrix{1,1}(Σ)))
        data = (;y=SVector.(y), u=SVector.(u))
        rts = RtsSmoother(model, data)
        rts(θ)
        @test log_likelihood(rts) ≈ ll
        @test mf ≈ reinterpret(Float64, SVector.(mean.(filter_density(rts))))
        @test σf ≈ reinterpret(Float64, SVector.(var.(filter_density(rts))))
        @test mp ≈ reinterpret(Float64, SVector.(mean.(predictive_density(rts))))
        @test σp ≈ reinterpret(Float64, SVector.(var.(predictive_density(rts))))
        @test ms ≈ reinterpret(Float64, SVector.(mean.(smoothing_density(rts))))
        @test σs ≈ reinterpret(Float64, SVector.(var.(smoothing_density(rts))))
    end

    @testset "2D-example" begin
        y,u,A,B,C,D,Q,R,μ,Σ,mf,Σf,mp,Σp,ms,Σs,ll = get_2D_rts()
        model = LinearGaussian{2,1,1}()
        θ = LinearGaussianPar(
            A=SMatrix{2,2}(A),
            B=SMatrix{2,1}(B),
            C=SMatrix{1,2}(C),
            D=SMatrix{1,1}(D),
            Q=PDMat(SMatrix{2,2}(Q)),
            R=PDMat(SMatrix{1,1}(R)),
            μ₀=SVector{2}(μ),
            Σ₀=PDMat(SMatrix{2,2}(Σ)))
        data = (;y=SVector.(y), u=SVector.(u))
        rts = RtsSmoother(model, data)
        rts(θ)
        @test log_likelihood(rts) ≈ ll
        @test mf ≈ reinterpret(reshape, Float64, SVector.(mean.(filter_density(rts))))
        @test Σf[:] ≈ reinterpret(Float64, SVector.(cov.(filter_density(rts))))
        @test mp ≈ reinterpret(reshape, Float64, SVector.(mean.(predictive_density(rts))))
        @test Σp[:] ≈ reinterpret(Float64, SVector.(cov.(predictive_density(rts))))
        @test ms ≈ reinterpret(reshape, Float64, SVector.(mean.(smoothing_density(rts))))
        @test Σs[:] ≈ reinterpret(Float64, SVector.(cov.(smoothing_density(rts))))
    end
end
