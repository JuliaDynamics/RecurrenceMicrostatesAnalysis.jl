export Parameter, optimize

##########################################################################################
#   Parameter
##########################################################################################
"""
    Parameter

Abstract supertype for free parameters that can be optimized using **Recurrence Microstates Analysis**.

## Implementations
- [`Threshold`](@ref)
"""
abstract type Parameter end

##########################################################################################
#   Implementation: optimize
##########################################################################################
"""
    optimize(param::Parameter, args...)

Optimize a free [`Parameter`](@ref) using a specific complexity measure.

!!! warning "Performance"
    The `optimize` function may compute multiple distributions and can be computationally expensive.
    Avoid calling it inside performance-critical loops.
"""
function optimize(param::Parameter)
    T = typeof(param)
    msg = "`optimize` not implemented without arguments for $T."
    throw(ArgumentError(msg))
end

##########################################################################################