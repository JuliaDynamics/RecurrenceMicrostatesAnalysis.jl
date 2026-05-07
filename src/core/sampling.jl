export SamplingMode, AbstractSamplingSpace, RectSamplingSpace, TriangleSamplingSpace, ColumnSamplingSpace

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
abstract type InternalSamplingSpace end

"""
    AbstractSamplingSpace

Abstract supertype defining from where the package must sample the initial positions
\$(i, j)\$ of each microstate.

# Implementations
- [`RectSamplingSpace`](@ref)
- [`TriangleSamplingSpace`](@ref)
- [`ColumnSamplingSpace`](@ref)
"""
abstract type AbstractSamplingSpace end
#.........................................................................................
#   Abstract sampling spaces
#.........................................................................................
"""
    RectSamplingSpace <: AbstractSamplingSpace

This sampling space defines that a recurrence microstate can be retrived from the all
possible indexes.
"""
struct RectSamplingSpace <: AbstractSamplingSpace end

"""
    TriangleSamplingSpace <: AbstractSamplingSpace

This sampling space defines that a recurrence microstate can be only retrived from
the part ahead the line of identity of a recurrence plot. It means, \$ i < j \$ for
\$ j \\leq K - N + 1\$.
"""
struct TriangleSamplingSpace <: AbstractSamplingSpace end

"""
    ColumnSamplingSpace <: AbstractSamplingSpace

This sampling space defines that a recurrence microstate can be only retrived from
a specific column \$I\$. It means that \$ i = I \$ and \$ j \\in [1, K - N + 1] \$.

# Constructor
```julia
ss = ColumnSamplingSpace(I)
```

Note that `I` must be lesser than `K - N + 1`, being `N` the microstate size.
"""
struct ColumnSamplingSpace <: AbstractSamplingSpace
    I::Int
end

#.........................................................................................
#   Internal sampling spaces
#.........................................................................................
struct SSRect2 <: InternalSamplingSpace
    W::Int
    H::Int
end

struct SSRectN{D} <: InternalSamplingSpace
    space::NTuple{D, Int}
end

struct SSTriangle <: InternalSamplingSpace
    N::Int
    L::Int
end

struct SSColumn <: InternalSamplingSpace
    I::Int
    H::Int
end
#.........................................................................................
SamplingSpace(
    shape::MicrostateShape,
    abs::AbstractSamplingSpace,
    x::Union{StateSpaceSet, AbstractGPUVector{<: SVector}}, 
    y::Union{StateSpaceSet, AbstractGPUVector{<: SVector}}
) = throw(ArgumentError("`InternalSamplingSpace` not implemented for a microstate shape $(typeof(shape)), abstract sampling space $(typeof(abs)), and input data types $(typeof(x)) for `x`, and $(typeof(y)) for `y`."))

SamplingSpace(
    shape::MicrostateShape, 
    abs::AbstractSamplingSpace,
    x::AbstractArray{<: Real}, 
    y::AbstractArray{<: Real}
) = throw(ArgumentError("`InternalSamplingSpace` not implemented for a microstate shape $(typeof(shape)), abstract sampling space $(typeof(abs)), and input data types $(typeof(x)) for `x`, and $(typeof(y)) for `y`."))

##########################################################################################
#   Implementation: sampling
##########################################################################################
function get_sample(
    core::RMACore,
    mode::SamplingMode,
    space::InternalSamplingSpace
)
    msg = "`get_sample` not implemented for core $(typeof(core)), sampling mode $(typeof(mode)), and sampling space $(typeof(space))"
    throw(ArgumentError(msg))
end

##########################################################################################
#   Utils: number of samples
##########################################################################################
function get_num_samples(
    mode::SamplingMode,
    space::InternalSamplingSpace
)
    msg = "`get_num_samples` not implemented for sampling mode $(typeof(mode)), and sampling space $(typeof(space))"
    throw(ArgumentError(msg))
end

##########################################################################################