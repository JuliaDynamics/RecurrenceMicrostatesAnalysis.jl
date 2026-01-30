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
using ComplexityMeasures
using Distances
using GPUArraysCore
using KernelAbstractions
using Random
using Reexport
using StaticArrays

@reexport using Adapt
@reexport using StateSpaceSets

const DEFAULT_METRIC = Euclidean()

##########################################################################################
#   Core API types and functions
##########################################################################################
include("core/recurrence_microstates.jl")

include("core/abstract_core.jl")
include("core/recurrence.jl")
include("core/shape.jl")
include("core/sampling.jl")

include("core/cpu_core.jl")

include("core/gpu/gpu_metric.jl")
include("core/gpu/gpu_core.jl")

include("core/measures.jl")
include("core/optimize.jl")
include("core/operation.jl")

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