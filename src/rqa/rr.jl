export RecurrenceRate

##########################################################################################
#   Quantification Measure: RecurrenceRate
##########################################################################################
"""
    RecurrenceRate <: QuantificationMeasure

Define the *Recurrence Rate* (RR) quantification measure.

RR can be computed either from a distribution of recurrence microstates or directly from
time-series data. In both cases, the computation is performed via the [`measure`](@ref)
function.

#   Using a distribution
```julia
measure(::RecurrenceRate, dist::Probabilities)
```

##  Arguments
- `dist`: A distribution of recurrence microstates.

##  Returns
A `Float64` corresponding to the estimated recurrence rate.

##  Examples
```julia
using RecurrenceMicrostatesAnalysis, Distributions
data = StateSpaceSet(rand(Uniform(0, 1), 1000))
dist = distribution(data, 0.27, 3)
rr = measure(RecurrenceRate(), dist)
```

#   Using a time series
```julia
measure(::RecurrenceRate, [x]; kwargs...)
```
##  Arguments
- `[x]`: Time-series data provided as a [`StateSpaceSet`](@ref).

##  Returns
A `Float64` corresponding to the estimated recurrence rate.

##  Keyword Arguments
- `N`: Integer defining the microstate size. The default value is `3`.
- `threshold`: Threshold used to compute the RMA distribution. By default, this is chosen as
    the threshold that maximizes the recurrence microstate entropy (RME).

##  Examples
```julia
using RecurrenceMicrostatesAnalysis, Distributions
data = StateSpaceSet(rand(Uniform(0, 1), 1000))
rme = measure(RecurrenceRate(), data; N = 4)
```
"""
struct RecurrenceRate <: QuantificationMeasure end

##########################################################################################
#   Implementation: measure
##########################################################################################
#       Using as input a RMA distribution.
#.........................................................................................
function measure(
        ::RecurrenceRate, 
        dist::Probabilities
    )
    result = 0.0
    hv = Int(log2(length(dist)))

    for i in eachindex(dist)
        rr = count_ones(i - 1) / hv
        result += rr * dist[i]
    end

    return result
end
#.........................................................................................
#       Using as input a time series
#.........................................................................................
function measure(
        ::RecurrenceRate, 
        x::Union{StateSpaceSet, Vector{<:Real}}, 
        y::Union{StateSpaceSet, Vector{<:Real}} = x; 
        n::Integer = 3, 
        threshold::Real = optimize(Threshold(), RecurrenceEntropy(), x, n)[1],
        metric::Metric = DEFAULT_METRIC
    )
    dist = probabilities(RecurrenceMicrostates(threshold, n; metric = metric), x, y)
    return measure(RecurrenceRate(), dist)
end

##########################################################################################