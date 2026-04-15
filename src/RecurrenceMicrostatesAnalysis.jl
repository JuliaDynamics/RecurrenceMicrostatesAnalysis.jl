module RecurrenceMicrostatesAnalysis

@doc let 
    path = joinpath(dirname(@__DIR__), "README.md")
    include_dependency(path)
    read(path, String)
end RecurrenceMicrostatesAnalysis

##########################################################################################
#   Packages and constants
##########################################################################################
using Atomix
using Combinatorics
using Distances
using GPUArraysCore
using KernelAbstractions
using Random
using Reexport
using StaticArrays
using Statistics

@reexport using ComplexityMeasures
@reexport using StateSpaceSets

const DEFAULT_METRIC = Euclidean()

##########################################################################################
#   Core API types and functions
##########################################################################################
include("core/abstract_core.jl")
include("core/recurrence.jl")
include("core/shape.jl")
include("core/sampling.jl")
include("core/gpu/gpu_metric.jl")
include("core/recurrence_microstates.jl")

include("core/cpu_core.jl")
include("core/gpu/gpu_core.jl")

include("core/optimize.jl")
include("core/operation.jl")

include("core/complexity_measures_interface.jl")

##########################################################################################
#   Recurrence functions, motif shapes, and sampling modes
##########################################################################################
include("recurrences/recurrences.jl")
include("shapes/shapes.jl")
include("sampling/sampling.jl")

##########################################################################################
#   Quantifiers
##########################################################################################
include("rqa/rqa.jl")

##########################################################################################
#   Utils
##########################################################################################
include("utils/gpu_metrics/gpu_metrics.jl")
include("utils/opt/opt.jl")
include("utils/operations/ops.jl")

end