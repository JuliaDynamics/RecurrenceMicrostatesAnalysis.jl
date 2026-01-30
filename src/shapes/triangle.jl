export TriangleMicrostate

##########################################################################################
#   MicrostateShape: Triangle + Constructors and sub-types
##########################################################################################
"""
    TriangleMicrostate{N, B} <: MicrostateShape

TriangleMicrostate{N, B} <: MicrostateShape

Define a triangular microstate shape, originally introduced by Hirata in 2021
[Hirata2021Triangle](@cite).

#   Constructor
```julia
TriangleMicrostate(N::Int; B::Int = 2)
```
where `N` defines the size of the triangular microstate.

#   Example
```julia
N = 3
triangle = TriangleMicrostate(N)
```
The corresponding microstate structure is given by:
```math
\\begin{pmatrix}
R_{i,j} & R_{i,j + 1} & R_{i,j + 2} \\\\
 & R_{i + 1,j + 1} & R_{i + 1,j + 2} \\\\
 & & R_{i + 2,j + 2} \\\\
\\end{pmatrix}
```

!!! compat
    Triangular microstate shape is not compatible with spatial data.
"""
struct TriangleMicrostate{N, B} <: MicrostateShape end

TriangleMicrostate(N::Int; B::Int = 2) = TriangleMicrostate{N, B}()

##########################################################################################
#   Implementations: SamplingSpace
##########################################################################################
#   Based on time series: (CPU & GPU)
#.........................................................................................
SamplingSpace(
    ::TriangleMicrostate{N, B}, 
    x::Union{StateSpaceSet, AbstractGPUVector{SVector{D, Float32}}}, 
    y::Union{StateSpaceSet, AbstractGPUVector{SVector{D, Float32}}}
) where {N, B, D} = SSRect2(length(x) - N + 1, length(y) - N + 1)

##########################################################################################
#   Implementations: utils — histogram size, power vector, and offsets
##########################################################################################
@generated function get_histogram_size(::TriangleMicrostate{N, B}) where {N, B}
    size = B^((N * (N + 1)) ÷ 2)
    return :( $size )
end

@generated function get_power_vector(::CPUCore, ::TriangleMicrostate{N, B}) where {N, B}
    expr = :(SVector{$(N*(N + 1) ÷ 2)}( $([:(B^$( (((j - 1) * j) ÷ 2) + (i - 1) )) for j in 1:N for i in 1:j]... ) ))
    return expr
end

@generated function get_offsets(::CPUCore, ::TriangleMicrostate{N, B}) where {N, B}
    elems = [ :(SVector{2, Int}($i, $j)) for j in 0:(N - 1) for i in 0:j]
    return :( SVector{$(N*(N + 1) ÷ 2), $(SVector{2, Int})}( $(elems...) ) )
end

@generated function get_power_vector(::GPUCore, ::TriangleMicrostate{N, B}) where {N, B}
    expr = :(SVector{$(N*(N + 1) ÷ 2)}( $([:(Int32(B^$( (((j - 1) * j) ÷ 2) + (i - 1) ))) for j in 1:N for i in 1:j]... ) ))
    return expr
end

@generated function get_offsets(::GPUCore, ::TriangleMicrostate{N, B}) where {N, B}
    elems = [ :(SVector{2, Int32}($(Int32(i)), $(Int32(j)))) for j in 0:(N - 1) for i in 0:j]
    return :( SVector{$(N*(N + 1) ÷ 2), $(SVector{2, Int32})}( $(elems...) ) )
end