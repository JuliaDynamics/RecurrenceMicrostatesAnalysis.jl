export CorridorRecurrence

##########################################################################################
#   RecurrenceExpression + Constructors
##########################################################################################
"""
    CorridorRecurrence <: RecurrenceExpression

Recurrence expression defined by the corridor criterion introduced in
[Iwanski1998Corridor](@cite):
```math
r_{(i, j)} = \\Theta(|\\vec{x}_i - \\vec{y}_j| - \\varepsilon_{min}) \\cdot \\Theta(\\varepsilon_{max} - |\\vec{x}_i - \\vec{y}_j|),
```
where \$\\Theta(\\cdot)\$ denotes the Heaviside function and \$(\\varepsilon_{\\min}, \\varepsilon_{\\max})\$
define the minimum and maximum distance thresholds for two states to be considered recurrent.

The `CorridorRecurrence` struct stores the corridor thresholds `ε_min` and `ε_max`, as well as the
distance `metric` used to evaluate \$\\|\\vec{x}_i - \\vec{y}_j\\|\$. The metric must be defined using
the **Distances.jl** package.

If the data for \$\\vec{x}\$ and \$\\vec{y}\$ are the same, the result is a recurrence plot;
otherwise, it is a cross-recurrence plot.

#   Constructor
```julia
CorridorRecurrence(ε_min::Real, ε_max::Real; metric::Metric = Euclidean())
```

#   Examples
```julia
CorridorRecurrence(0.05, 0.27)
CorridorRecurrence(0.05, 0.27; metric = Cityblock())
CorridorRecurrence(0.05, 0.27; metric = GPUEuclidean())
```

The recurrence evaluation is performed via the [`recurrence`](@ref) function.
For GPU execution, the corresponding implementation is provided by `gpu_recurrence`.
"""
struct CorridorRecurrence{T <: Real, M <: Metric} <: RecurrenceExpression{T, M}
    ε_min::T
    ε_max::T
    metric::M
end
#.........................................................................................
function CorridorRecurrence(ε_min::Real, ε_max::Real; metric::Metric = DEFAULT_METRIC)
    @assert ε_min >= 0 throw(ArgumentError("threshold must be greater than zero."))
    @assert ε_min < ε_max throw(ArgumentError("ε_min must be less than ε_max."))
    return CorridorRecurrence(ε_min, ε_max, metric)
end

##########################################################################################
#   Implementations
##########################################################################################
#   Based on time series: (CPU)
#.........................................................................................
@inline function recurrence(
    expr::CorridorRecurrence,
    x::StateSpaceSet,
    y::StateSpaceSet,
    i::Int,
    j::Int,
)
    distance = @inbounds evaluate(expr.metric, x.data[i], y.data[j])
    return UInt8(distance ≥ expr.ε_min && distance ≤ expr.ε_max)
end
#.........................................................................................
#   Based on time series: (GPU)
#.........................................................................................
@inline function gpu_recurrence(expr::CorridorRecurrence, x, y, i, j, n)
    distance = gpu_evaluate(expr.metric, x, y, i, j, n)
    return UInt8(distance ≥ expr.ε_min && distance ≤ expr.ε_max)
end
#.........................................................................................
#   Based on spatial data: (CPU only)
#.........................................................................................
@inline function recurrence(
    expr::CorridorRecurrence,
    x::AbstractArray{<:Real},
    y::AbstractArray{<:Real},
    i::NTuple{N, Int}, 
    j::NTuple{M, Int},
) where {N, M} 
    distance = @inbounds evaluate(expr.metric, view(x, :, i...), view(y, :, j...))
    return UInt8(distance ≥ expr.ε_min && distance ≤ expr.ε_max)
end

##########################################################################################