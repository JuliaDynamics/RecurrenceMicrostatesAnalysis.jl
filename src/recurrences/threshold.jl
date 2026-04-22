export ThresholdRecurrence

##########################################################################################
#   RecurrenceExpression + Constructors
##########################################################################################
"""
    ThresholdRecurrence <: RecurrenceExpression

Recurrence expression defined by the standard threshold criterion:
```math
r_{(i,j)} = \\Theta(\\varepsilon - |\\vec{x}_i - \\vec{y}_j|),
```
where \$\\Theta(\\cdot)\$ denotes the Heaviside function and \$\\varepsilon\$ is the distance
threshold defining the maximum separation for two states to be considered recurrent.

The `ThresholdRecurrence` struct stores the threshold parameter `ε`, as well as the distance
`metric` used to evaluate \$\\|\\vec{x} - \\vec{y}\\|\$. The metric must be defined using the
**Distances.jl** package.

If the data for \$\\vec{x}\$ and \$\\vec{y}\$ are the same, the result is a recurrence plot;
otherwise, it is a cross-recurrence plot.

## Constructor
```julia
ThresholdRecurrence(ε::Real; metric::Metric = Euclidean())
```

## Examples
```julia
ThresholdRecurrence(0.27)
ThresholdRecurrence(0.27; metric = Cityblock())
ThresholdRecurrence(0.27; metric = GPUEuclidean())
```

The recurrence evaluation is performed via the [`recurrence`](@ref) function.
For GPU execution, the corresponding implementation is provided by `gpu_recurrence`.
"""
struct ThresholdRecurrence{T <: Real, M <: Metric} <: RecurrenceExpression{T, M}
    ε::T
    metric::M
end
#.........................................................................................
function ThresholdRecurrence(ε::Real; metric::Metric = DEFAULT_METRIC)
    @assert ε >= 0 throw(ArgumentError("Threshold must be greater than zero."))
    return ThresholdRecurrence(ε, metric)
end

##########################################################################################
#   Implementations
##########################################################################################
#   Based on time series: (CPU)
#.........................................................................................
@inline function recurrence(
    expr::ThresholdRecurrence,
    x::StateSpaceSet,
    y::StateSpaceSet,
    i::Int,
    j::Int,
)
    distance = @inbounds evaluate(expr.metric, x.data[i], y.data[j])
    return UInt8(distance ≤ expr.ε)
end
#.........................................................................................
#   Based on time series: (GPU)
#.........................................................................................
@inline function gpu_recurrence(expr::ThresholdRecurrence, x, y, i, j, n)
    distance = gpu_evaluate(expr.metric, x, y, i, j, n)
    return UInt8(distance ≤ expr.ε)
end
#.........................................................................................
#   Based on spatial data: (CPU only)
#.........................................................................................
@inline function recurrence(
    expr::ThresholdRecurrence,
    x::AbstractArray{<:Real},
    y::AbstractArray{<:Real},
    i::NTuple{N, Int}, 
    j::NTuple{M, Int},
) where {N, M} 
    distance = @inbounds evaluate(expr.metric, view(x, :, i...), view(y, :, j...))
    return UInt8(distance ≤ expr.ε)
end

##########################################################################################