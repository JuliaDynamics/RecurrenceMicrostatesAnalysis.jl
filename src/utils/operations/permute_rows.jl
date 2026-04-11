export PermuteRows

##########################################################################################
#   Permute Rows
##########################################################################################
"""
    PermuteRows{R, C} <: Operation

Operation that permutes the rows of a microstate \$\\mathbf{M}\$.

To initialize a `PermuteRows` operation, a rectangular microstate shape must be
provided via a [`RectMicrostate`](@ref) structure:
```julia
PermuteRows(::Rect2Microstate{R, C, B})
```

#   Examples
```julia
PermuteRows(RectMicrostate(3, 3))     #   Microstate 3 x 3
PermuteRows(RectMicrostate(3, 1))     #   Microstate 3 x 1 (it is a column)
```

This operation is applied via the [`operate`](@ref) function:
```julia
operate(::PermuteRows, I::Int, σ::Vector{Int})
```

#   Arguments
- `op`: A `PermuteRows` operation.
- `I`: Decimal identifier of the microstate (1-based).
- `σ`: Permutation of rows to be applied.

#   Returns
The resulting microstate binary identifier (1-based).
"""
struct PermuteRows{R, C} <: Operation end

PermuteRows(::Rect2Microstate{R, C, B}) where {R, C, B} = PermuteRows{R, C}()

##########################################################################################
#   Operate a permutation of rows
##########################################################################################
function operate(::PermuteRows{R, C}, I::Int, σ::Vector{Int}) where {R, C}

    B = UInt32(I - 1)
    res_b = UInt32(0)
    mask = UInt32(2^C) - one(UInt32)

    for n ∈ 0:R - 1
        res_b |= ((B >> (n * C)) & mask) << ((σ[n + 1] - 1) * C)
    end

    return Int(res_b) + 1
end

##########################################################################################