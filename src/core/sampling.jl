export SamplingMode, SamplingSpace

##########################################################################################
#   Sampling Mode
##########################################################################################
"""
    SamplingMode

Abstract supertype defining how the initial position \$(i, j)\$ of each microstate is selected
during the construction of recurrence microstate distributions.

# Implementations
- [`SRandom`](@ref)
- [`Full`](@ref)
"""
abstract type SamplingMode end

##########################################################################################
#   Sampling Space
##########################################################################################
"""
    SamplingSpace

Define the range of valid indices used to sample the initial positions \$(i, j)\$ of microstates.
"""
abstract type SamplingSpace end
#.........................................................................................
#   Based on time series: RP & CRP (CPU)
#.........................................................................................
struct SSRect2 <: SamplingSpace
    W::Int
    H::Int
end
#.........................................................................................
SamplingSpace(
    shape::MicrostateShape, 
    x::Union{StateSpaceSet, AbstractGPUVector{<: SVector}}, 
    y::Union{StateSpaceSet, AbstractGPUVector{<: SVector}}
) = throw(ArgumentError("`SamplingSpace` not implemented for a microstate shape $(typeof(shape)), and input data types $(typeof(x)) for `x`, and $(typeof(y)) for `y`."))
#.........................................................................................
#   Based on spatial data: SRP & CSRP (CPU)
#.........................................................................................
struct SSRectN{D} <: SamplingSpace
    space::NTuple{D, Int}
end
#.........................................................................................
SamplingSpace(
    shape::MicrostateShape, 
    x::AbstractArray{<: Real}, 
    y::AbstractArray{<: Real}
) = throw(ArgumentError("`SamplingSpace` not implemented for a microstate shape $(typeof(shape)), and input data types $(typeof(x)) for `x`, and $(typeof(y)) for `y`."))

##########################################################################################
#   Implementation: sampling
##########################################################################################
function get_sample(
    core::RMACore,
    mode::SamplingMode,
    space::SamplingSpace
)
    msg = "`get_sample` not implemented for core $(typeof(core)), sampling mode $(typeof(mode)), and sampling space $(typeof(space))"
    throw(ArgumentError(msg))
end

##########################################################################################
#   Utils: number of samples
##########################################################################################
function get_num_samples(
    mode::SamplingMode,
    space::SamplingSpace
)
    msg = "`get_num_samples` not implemented for sampling mode $(typeof(mode)), and sampling space $(typeof(space))"
    throw(ArgumentError(msg))
end

##########################################################################################