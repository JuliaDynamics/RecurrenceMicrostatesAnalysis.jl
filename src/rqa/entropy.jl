export RecurrenceEntropy

##########################################################################################
#   Quantification Measure: RecurrenceRate
##########################################################################################
"""
    RecurrenceEntropy <: QuantificationMeasure

Define the *Recurrence Microstates Entropy* (RME) quantification measure [Corso2018Entropy](@cite).

RME can be computed either from a distribution of recurrence microstates or directly from
time-series data. In both cases, the computation is performed via the [`measure`](@ref)
function.

#   Using a distribution
```julia
measure(::RecurrenceEntropy, dist::Probabilities)
```
##  Arguments
- `dist`: A distribution of recurrence microstates.

##  Returns
A `Float64` corresponding to the RME computed using the Shannon entropy.

### Examples
```julia
using RecurrenceMicrostatesAnalysis, Distributions
data = StateSpaceSet(rand(Uniform(0, 1), 1000))
dist = distribution(data, 0.27, 3)
rme = measure(RecurrenceEntropy(), dist)
```

#   Using a time series
```julia
measure(::RecurrenceEntropy, [x]; kwargs...)
```
##  Arguments
- `[x]`: Time-series data provided as an [`StateSpaceSet`](@ref).

##  Returns
A `Float64` corresponding to the **maximum** RME computed using the Shannon entropy.

##  Keyword Arguments
- `N`: Integer defining the microstate size. The default value is `3`.

### Examples
```julia
using RecurrenceMicrostatesAnalysis, Distributions
data = StateSpaceSet(rand(Uniform(0, 1), 1000))
rme = measure(RecurrenceEntropy(), data; N = 4)
```
"""
struct RecurrenceEntropy <: QuantificationMeasure end

##########################################################################################
#   Implementation: measure
##########################################################################################
#       Using as input a RMA distribution.
#.........................................................................................
function measure(::RecurrenceEntropy, dist::Probabilities)
    return entropy(Shannon(; base = MathConstants.e), dist)
end
#.........................................................................................
#       Using as input a time series
#.........................................................................................
function measure(::RecurrenceEntropy, x::Union{StateSpaceSet, Vector{<:Real}}; N::Integer = 3)
    return optimize(Threshold(), RecurrenceEntropy(), x, N)[2]
end

##########################################################################################