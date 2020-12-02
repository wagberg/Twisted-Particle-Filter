__precompile__(true)
module GaussianSmoothers

using Distributions
using LinearAlgebra
using StaticArrays
using ForwardDiff
using Random
import Random: rand

# Kalman, Extended Kalman, Unscented Kalman Filters

export
	DynamicsModel,
	LinearDynamicsModel,
	NonlinearDynamicsModel,
	ObservationModel,
	LinearObservationModel,
	NonlinearObservationModel,
	predict,
	measure
include("models.jl")


export
	AbstractFilter,
	AbstractSmoother,
	KalmanFilter,
	ExtendedKalmanFilter,
	UnscentedKalmanFilter,
	GaussianBelief,
	RtsSmoother,
	ExtendedRtsSmoother
include("kf_classes.jl")

export 
	simulate_step,
	simulate,
	run_filter,
	run_smoother,
	likelihood,
	unpack
include("simulate.jl")

export	update
include("kf.jl")
include("ekf.jl")

export
	unscented_transform,
	unscented_transform_inverse
include("ukf.jl")

# Utilities
export
	belief_ellipse
include("utils.jl")

end # module
