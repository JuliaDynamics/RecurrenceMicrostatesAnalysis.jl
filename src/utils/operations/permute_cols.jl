export PermuteColumns

##########################################################################################
#   Permute Columns
##########################################################################################
"""
    PermuteColumns{R, C} <: Operation

Operation that permutes the columns of a microstate \$\\mathbf{M}\$.

To initialize a `PermuteColumns` operation, a rectangular microstate shape must be
provided via a [`Rect`](@ref) structure:
```julia
PermuteColumns(::Rect2Microstate{R, C, B}; S::Vector{Vector{Int}} = collect(permutations(1:C))
```
Here, the keyword argument `S` defines the set \$S_n\$ of column permutations. The
`PermuteColumns` struct precomputes the column permutations for each row of the microstate.
These precomputed permutations can be accessed via the field `Q`.

#   Examples
```julia
PermuteColumns(RectMicrostate(3, 3))     #   Microstate 3 x 3
PermuteColumns(RectMicrostate(1, 3))     #   Microstate 1 x 3 (it is a line)
```

This operation is applied via the [`operate`](@ref) function:
```julia
operate(op::PermuteColumns, I::Int, Qi::Int)
```

#   Arguments
- `op`: A `PermuteColumns` operation.
- `I`: Decimal identifier of the microstate (1-based).
- `Qi`: Index of the permutation in the set `S`.

#   Returns
The resulting microstate decimal identifier (1-based).
"""
struct PermuteColumns{R, C} <: Operation
    Q::Vector{Vector{UInt32}}
end

PermuteColumns(
        ::Rect2Microstate{R, C, B};
        S::Vector{Vector{Int}} = collect(permutations(1:C))
    ) where {R, C, B} = PermuteColumns{R, C}(precompute_Q(R, C, S))

##########################################################################################
#   Operate a permutation of columns
##########################################################################################
function operate(op::PermuteColumns{R, C}, I::Int, Qi::Int) where {R, C}
    B = UInt32(I - 1)
    res_b = UInt32(0)
    mask = UInt32(2^C) - one(UInt32)
    S = op.Q[Qi]

    for n ∈ 0:R - 1
        res_b |= (S[Int((B >> (n * C)) & mask) + 1]) << (n * C)
    end

    return Int(res_b) + 1
end

##########################################################################################
#   Precompute Q
##########################################################################################
function precompute_Q(rows::Int, columns::Int, S::Vector{Vector{Int}})
    Q = Vector{Vector{UInt32}}(undef, length(S))
    B = Vector{UInt32}(undef, 2^(rows))

    for i ∈ eachindex(B)
        B[i] = UInt32(i - 1)
    end

    for i ∈ eachindex(S)
        Q[i] = Vector{UInt32}(undef, length(B))
        for n ∈ eachindex(B)
            b = zero(UInt32)
            for c ∈ 0:columns - 1
                b |= ((B[n] >> c) & 1) << (S[i][c + 1] - 1)
            end
            Q[i][n] = b
        end
    end

    return Q
end

##########################################################################################