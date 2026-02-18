export Transpose

##########################################################################################
#   Transpose the microstates
##########################################################################################
"""
    Transpose{R, C} <: Operation

Operation that transposes a microstate \$\\mathbf{M}\$.

To initialize a `Transpose` operation, a rectangular microstate shape must be
provided via a [`Rect`](@ref) structure:
```julia
Transpose(::Rect2Microstate{R, C, B})
```

#   Examples
```julia
Transpose(RectMicrostate(3, 3))     # 3 x 3 microstate
```

This operation is applied via the [`operate`](@ref) function:
```julia
operate(::Transpose, I::Int)
```
#   Arguments
- `op`: A `Transpose` operation.
- `I`: DEcima identifier of the microstate (1-based).

#   Returns
The resulting microstate decimal identifier (1-based).
"""
struct Transpose{R, C} <: Operation end

Transpose(::Rect2Microstate{R, C, B}) where {R, C, B} = Transpose{R, C}()

##########################################################################################
#   Operate a transposition
##########################################################################################
function operate(::Transpose{R, C}, I::Int) where {R, C}
    B = UInt32(I - 1)
    res_b = UInt32(0)

    for r ∈ 0:R - 1
        for c ∈ 0:C - 1
            res_b |= ((B >> ((r * R) + c)) & 1) << ((c * R) + r)
        end
    end

    return Int(res_b) + 1
end

##########################################################################################