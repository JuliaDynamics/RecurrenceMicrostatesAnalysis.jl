export RecurrenceRate

##########################################################################################
#   Quantification Measure: RecurrenceRate
#   Complexity Measure Implementation
##########################################################################################
"""
    RecurrenceRate <: ComplexityEstimator
    RecurrenceRate(ε::Float64, N::Int = 1; kwargs...)

An estimator of the recurrence rate, used with [complexity](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/complexity/#ComplexityMeasures.complexity).
It uses a \$1 \\times 1\$ microstate by default, but you can set a different size
via the `N` parameter. The recurrence rate is estimated for a threshold `ε`.

## Keyword arguments
- `metric::Metric`: The metric used to compute recurrence.
- `ratio`: The sampling ratio. The default is `0.1`.
- `sampling`: The sampling mode. The default is [`SRandom`](@ref).

## Description

Recurrence rate (RR) is defined as [Webber2015Recurrence](@cite)
```math
RR = \\frac{1}{K^2}\\sum_{i,j=1}^{K} r_{(i,j)}.
```

When estimating it using RMA, the recurrence rate is defined as the expected value
over the microstate distribution:
```math
RR \\approx \\sum_{i=1}^{2^\\sigma} p_i^{(N)}RR_i^{(N)},
```
where \$RR_i^{(N)}\$ denotes the recurrence rate of the \$i\$-th microstate.

!!! note "Performance"
    Although estimating RR using RMA is faster than typical RQA computation,
    the precision depends on the time series length. Therefore, for small time series,
    i.e., \$K \\leq 1000\$, we strongly recommend using standard RQA with
    [RecurrenceAnalysis.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/recurrenceanalysis/stable/quantification/#RecurrenceAnalysis.recurrencerate).
"""
struct RecurrenceRate{N, M, SM} <: ComplexityEstimator
    ε::Real
    metric::M
    sampling::SM
end

function complexity(
        c::RecurrenceRate{N},
        x::Union{StateSpaceSet, Vector{<:Real}, <:AbstractGPUVector{<:SVector}}, 
        y::Union{StateSpaceSet, Vector{<:Real}, <:AbstractGPUVector{<:SVector}} = x;
    ) where {N}
    
    probs = probabilities(RecurrenceMicrostates(c.ε, N; metric = c.metric, sampling = c.sampling), x, y)
    return measure(c, probs)
end

# -- Constructors
RecurrenceRate(ε::Real, N::Int = 1; metric::M = DEFAULT_METRIC, ratio::Float64 = 0.1, sampling::SM = SRandom(ratio)) where {M, SM} = RecurrenceRate{N, M, SM}(ε, metric, sampling)

##########################################################################################
#   Internal: measure from probabilities
##########################################################################################
# This is an internal function which estimates the recurrence rate from a recurrence microstate
# outcome space, using a given probability distribution that was computed from this
# outcome space.

# This function works for any recurrence microstates probabilities.
##########################################################################################
function measure(::RecurrenceRate, probs::Probabilities)
    result = 0.0
    hv = Int(log2(length(probs)))

    for i in eachindex(probs)
        rr = count_ones(i - 1) / hv
        result += rr * probs[i]
    end

    return result
end

##########################################################################################