export Laminarity

##########################################################################################
#   Quantification Measure: Laminarity
##########################################################################################
"""
    Laminarity <: QuantificationMeasure

Define the *Laminarity* (LAM) quantification measure.

LAM can be computed either from a distribution of recurrence microstates or directly from
time-series data. In both cases, the computation is performed via the [`measure`](@ref)
function.

#   Using a distribution
```julia
measure(::Laminarity, rmspace::RecurrenceMicrostates, dist::Probabilities)
```

##  Arguments
- `rmspace`: A recurrence outcome space.
- `dist`: A distribution of recurrence microstates. The distribution must be computed from
    **square** or **line** microstates of size 3.

##  Returns
A `Float64` corresponding to the estimated laminarity.

##  Examples
### Using square microstates
```julia
using RecurrenceMicrostatesAnalysis, Distributions
data = StateSpaceSet(rand(Uniform(0, 1), 1000))
rmspace = RecurrenceMicrostates(0.27, 3)
dist = probabilities(rmspace, data)
lam = measure(Laminarity(), dist)
```

### Using line microstates:
```julia
using RecurrenceMicrostatesAnalysis, Distributions
data = StateSpaceSet(rand(Uniform(0, 1), 1000))
rmspace = RecurrenceMicrostates(0.27, RectMicrostate(1, 3))
dist = probabilities(rmspace, data)
lam = measure(Laminarity(), dist)
```

# Using a time series
```julia
measure(::Laminarity, [x]; kwargs...)
```

##  Arguments
- `[x]`: Time-series data provided as a [`StateSpaceSet`](@ref).

##  Returns
A `Float64` corresponding to the estimated laminarity.

##  Keyword Arguments
- `threshold`: Threshold used to compute the RMA distribution. By default, this is chosen as
    the threshold that maximizes the recurrence microstate entropy (RME).

### Examples
```julia
using RecurrenceMicrostatesAnalysis, Distributions
data = StateSpaceSet(rand(Uniform(0, 1), 1000))
lam = measure(Laminarity(), data)
```

!!! note
    When time-series data are provided directly, RecurrenceMicrostatesAnalysis.jl uses
    line microstates by default.
"""
struct Laminarity <: QuantificationMeasure end

##########################################################################################
#   Implementation: measure
##########################################################################################
#       Using as input a RMA distribution.
#.........................................................................................
function measure(
        ::Laminarity, 
        rmspace::RecurrenceMicrostates, 
        dist::Probabilities
    )
    if (rmspace.shape isa Rect2Microstate{3, 3, 2} && length(dist) == 512)
        rr = measure(RecurrenceRate(), dist)

        values = zeros(Int, 64)
        v_idx = 1

        for a1 in 0:1, a2 in 0:1, a3 in 0:1, a4 in 0:1, a5 in 0:1, a6 in 0:1
            I_1 = 2 + 8 * a1 + 16 * a2 + 32 * a3 + 64 * a4 + 128 * a5 + 256 * a6
            values[v_idx] = I_1 + 1
            v_idx += 1
        end

        pl = 0.0
        for i in values
            pl += dist[i]
        end

        return 1 - ((1/rr) * pl)

   elseif (rmspace.shape isa Rect2Microstate{1, 3} && length(dist) == 8)
        rr = measure(RecurrenceRate(), dist)
        return 1 - ((1/rr) * dist[3])
    else
        msg = "Laminarity must be computed using square or line microstates with n = 3."
        throw(ArgumentError(msg))
    end
end
#.........................................................................................
#       Using as input a time series
#.........................................................................................
function measure(
    op::Laminarity, 
    x::Union{StateSpaceSet, Vector{<:Real}}, 
    y::Union{StateSpaceSet, Vector{<:Real}} = x; 
    threshold::Real = optimize(Threshold(), RecurrenceEntropy(), x, 3)[1],
    metric::Metric = DEFAULT_METRIC
    )
    rmspace = RecurrenceMicrostates(threshold, RectMicrostate(1, 3); metric = metric)
    dist = probabilities(rmspace, x, y)
    measure(op, rmspace, dist)
end

##########################################################################################