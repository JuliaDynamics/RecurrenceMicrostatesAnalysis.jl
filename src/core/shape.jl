export MicrostateShape

##########################################################################################
#   MicrostateShape
##########################################################################################
"""
    MicrostateShape

Abstract supertype defining the basic structure and layout of a microstate (or motif).

A `MicrostateShape` specifies which relative positions are retrieved from the data to evaluate
recurrences, and how these binary recurrence values are interpreted and mapped to a decimal
representation for counting.

All subtypes of `MicrostateShape` must include a field `expr`, which defines the
[`RecurrenceExpression`](@ref) used to compute recurrences.

# Implementations
- [`DiagonalMicrostate`](@ref)
- [`RectMicrostate`](@ref)
- [`TriangleMicrostate`](@ref)
"""
abstract type MicrostateShape end

##########################################################################################
#   Index computation
##########################################################################################
#   Based on spatial data: (CPU only)
#.........................................................................................
function compute_motif(
    shape::MicrostateShape,
    ::RecurrenceExpression,
    ::AbstractArray{<: Real},
    ::AbstractArray{<: Real},
    ::Vector{Int},
    ::Vector{Int},
    ::SVector{N, Int}
) where {N}
    msg = "`compute_motif` not implemented for spatial data and microstate shape $(typeof(shape))."
    throw(ArgumentError(msg))
end
##########################################################################################
#   Utils: number of recurrences and power vector.
##########################################################################################
function get_histogram_size(shape::MicrostateShape)
    T = typeof(shape)
    msg = "`get_histogram_size` not implemented for $T"
    throw(ArgumentError(msg))
end
#.........................................................................................
function get_power_vector(core::RMACore, shape::MicrostateShape)
    T = typeof(shape)
    msg = "`get_power_vector` not implemented for core $(typeof(core)), and microstate shape $T"
    throw(ArgumentError(msg))
end
#.........................................................................................
function get_offsets(core::RMACore, shape::MicrostateShape)
    T = typeof(shape)
    msg = "`get_offsets` not implemented for core $(typeof(core)), and microstate shape $T"
    throw(ArgumentError(msg))
end

##########################################################################################