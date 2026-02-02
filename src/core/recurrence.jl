export RecurrenceExpression, recurrence

##########################################################################################
#   Recurrence Expression
##########################################################################################
"""
    RecurrenceExpression

Abstract supertype for recurrence expressions implemented in the package.

Concrete subtypes of `RecurrenceExpression` must implement the [`recurrence`](@ref) function,
which defines how recurrence between two states is evaluated.

### Implementations
- [`Standard`](@ref)
- [`Corridor`](@ref)
"""
abstract type RecurrenceExpression{T, M} end

##########################################################################################
#   Function: recurrence
##########################################################################################
#   Based on time series: (CPU)
#.........................................................................................
"""
    recurrence(expr::RecurrenceExpression, [x], [y], [...])

Define how the recurrence state between `[x]` and `[y]` is computed for a given
[`RecurrenceExpression`](@ref).

The additional arguments (`[...]`) depend on whether the recurrence is computed for
time-series or spatial data.

### Time-series recurrence
```julia
function recurrence(expr::RecurrenceExpression, x::StateSpaceSet, y::StateSpaceSet, ::Int, ::Int)
```

The two `Int` arguments correspond to the positions \$(i, j)\$ in the time series used to
evaluate recurrence.

### Spatial recurrence
```julia
function recurrence(expr::RecurrenceExpression, x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, ::NTuple{N, Int}, ::NTuple{M, Int})
```

The two `NTuple{N, Int}` arguments correspond to the positions \$(\\vec i, \\vec j)\$ in the spatial data
used to evaluate recurrence.

!!! info
    To support GPU execution, recurrence expressions must implement `gpu_recurrence` instead of `recurrence`. 
    The arguments are equivalent, with the addition of the phase-space dimension
    as an input parameter. See [`Standard`](@ref) for a reference implementation.

### Output
The `recurrence` function must always return a `UInt8`: `0` for non-recurrence and `1` for recurrence.
"""
function recurrence(
    expr::RecurrenceExpression,
    x::StateSpaceSet,
    y::StateSpaceSet,
    ::Int,
    ::Int,
)
    T = typeof(expr)
    msg = "`recurrence` not implemented to $T and input data types $(typeof(x)) for `x` and $(typeof(y)) for `y`."
    throw(ArgumentError(msg))
end
#.........................................................................................
#   Based on spatial data: (CPU)
#.........................................................................................
function recurrence(
    expr::RecurrenceExpression,
    x::AbstractArray{<:Real},
    y::AbstractArray{<:Real},
    ::NTuple{N, Int}, 
    ::NTuple{M, Int},
) where {N, M} 
    T = typeof(expr)
    msg = "`recurrence` not implemented to $T and input data types $(typeof(x)) for `x` and $(typeof(y)) for `y`."
    throw(ArgumentError(msg))
end

##########################################################################################