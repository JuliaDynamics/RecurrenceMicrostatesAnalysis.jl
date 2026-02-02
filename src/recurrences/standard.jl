export Standard

##########################################################################################
#   RecurrenceExpression + Constructors
##########################################################################################
"""
    Standard <: RecurrenceExpression

Recurrence expression defined by the standard threshold criterion:
```math
R(\\vec{x}, \\vec{y}) = \\Theta(\\varepsilon - |\\vec{x} - \\vec{y}|),
```
where \$\\Theta(\\cdot)\$ denotes the Heaviside function and \$\\varepsilon\$ is the distance
threshold defining the maximum separation for two states to be considered recurrent.

The `Standard` struct stores the threshold parameter `ε`, as well as the distance `metric`
used to evaluate \$|\\vec{x} - \\vec{y}|\$. The metric must be defined using the [Distances.jl](https://github.com/JuliaStats/Distances.jl)
package.

#   Constructor
```julia
Standard(ε::Real; metric::Metric = Euclidean())
```

#   Examples
```julia
Standard(0.27)
Standard(0.27; metric = Cityblock())
```

The recurrence evaluation is performed via the [`recurrence`](@ref) function.
For GPU execution, the corresponding implementation is provided by `gpu_recurrence`.
"""
struct Standard{T <: Real, M <: Metric} <: RecurrenceExpression{T, M}
    ε::T
    metric::M
end
#.........................................................................................
function Standard(ε::Real; metric::Metric = DEFAULT_METRIC)
    @assert ε >= 0 throw(ArgumentError("threshold must be greater than zero."))
    return Standard(ε, metric)
end

##########################################################################################
#   Implementations
##########################################################################################
#   Based on time series: (CPU)
#.........................................................................................
@inline function recurrence(
    expr::Standard,
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
@inline function gpu_recurrence(expr::Standard, x, y, i, j, n)
    distance = gpu_evaluate(expr.metric, x, y, i, j, n)
    return UInt8(distance ≤ expr.ε)
end
#.........................................................................................
#   Based on spatial data: (CPU only)
#.........................................................................................
@inline function recurrence(
    expr::Standard,
    x::AbstractArray{<:Real},
    y::AbstractArray{<:Real},
    i::NTuple{N, Int}, 
    j::NTuple{M, Int},
) where {N, M} 
    distance = @inbounds evaluate(expr.metric, view(x, :, i...), view(y, :, j...))
    return UInt8(distance ≤ expr.ε)
end

##########################################################################################