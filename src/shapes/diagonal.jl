export DiagonalMicrostate

##########################################################################################
#   MicrostateShape: Diagonal + Constructors and sub-types
##########################################################################################
"""
    DiagonalMicrostate <: MicrostateShape

Define a diagonal microstate shape, which captures recurrences along diagonals of a
Recurrence Plot (RP).

#   Constructor
```julia
DiagonalMicrostate(N::Int; B::Int = 2)
```
where `N` defines the length of the diagonal microstate.

#   Example
```julia
diagonal = DiagonalMicrostate(expr, 3)
```

!!! info
    **DiagonalMicrostate** microstates are compatible with spatial data. However, they do not capture
    hyper-diagonals in Spatial Recurrence Plots (SRP). Only diagonals defined by sequential
    recurrences are supported, such as:
    ```math
    R_{i_1,i_2,j_1,j_2}, R_{i_1 + 1,i_2 + 1,j_1 + 1,j_2 + 1}, R_{i_1 + 2,i_2 + 2,j_1 + 2,j_2 + 2}, \\ldots, R_{i_1 + n - 1,i_2 + n - 1,j_1 + n - 1,j_2 + n - 1}
    ```
"""
struct DiagonalMicrostate{N, B} <: MicrostateShape end

DiagonalMicrostate(N::Int; B::Int = 2) = DiagonalMicrostate{N, B}(expr)

##########################################################################################
#   Implementations: SamplingSpace
##########################################################################################
#   Based on time series: (CPU & GPU)
#.........................................................................................
SamplingSpace(
    ::DiagonalMicrostate{N, B}, 
    x::Union{StateSpaceSet, AbstractGPUVector{SVector{D, Float32}}}, 
    y::Union{StateSpaceSet, AbstractGPUVector{SVector{D, Float32}}}
) where {N, B, D} = SSRect2(length(x) - N + 1, length(y) - N + 1)

function SamplingSpace(
    ::DiagonalMicrostate{N, B}, 
    x::AbstractArray{<: Real}, 
    y::AbstractArray{<: Real}
) where {N, B}

    dims_x = size(x)[2:end]
    dims_y = size(y)[2:end]

    dims = (dims_x..., dims_y...)
    
    space = ntuple(i -> dims[i] - N, length(dims))
    return SSRectN{length(dims)}(space)
end

##########################################################################################
#   Implementations: compute_motif (SRP)
##########################################################################################
@inline function compute_motif(
    ::DiagonalMicrostate,
    expr::RecurrenceExpression,
    x::AbstractArray{<: Real},
    y::AbstractArray{<: Real},
    idx::Vector{Int},
    itr::Vector{Int},
    power_vector::SVector{D, Int}
) where {D}
    
    index = 0
    dim_x = ndims(x) - 1
    dim_y = ndims(y) - 1

    copy!(itr, idx)

    @inbounds @fastmath for p in power_vector

        i = ntuple(k -> itr[k], dim_x)
        j = ntuple(k -> itr[dim_x + k], dim_y)

        index += recurrence(expr, x, y, i, j) * p

        itr .+= 1
    end

    return index + 1
end

##########################################################################################
#   Implementations: utils — histogram size, power vector, and offsets
##########################################################################################
@generated function get_histogram_size(::DiagonalMicrostate{N, B}) where {N, B}
    size = B^(N)
    return :( $size )
end

@generated function get_power_vector(::CPUCore, ::DiagonalMicrostate{N, B}) where {N, B}
    expr = :(SVector{$N}( $([:(B^$i) for i in 0:(N-1)]... ) ))
    return expr
end

@generated function get_offsets(::CPUCore, ::DiagonalMicrostate{N, B}) where {N, B}
    elems = [ :(SVector{2, Int}($n, $n)) for n in 0:(N - 1)]
    return :( SVector{$N, $(SVector{2, Int})}( $(elems...) ) )
end

@generated function get_power_vector(::GPUCore, ::DiagonalMicrostate{N, B}) where {N, B}
    expr = :(SVector{$N}( $([:(Int32(B^$i)) for i in 0:(N-1)]... ) ))
    return expr
end

@generated function get_offsets(::GPUCore, ::DiagonalMicrostate{N, B}) where {N, B}
    elems = [ :(SVector{2, Int32}($(Int32(n)), $(Int32(n)))) for n in 0:(N - 1)]
    return :( SVector{$N, $(SVector{2, Int32})}( $(elems...) ) )
end