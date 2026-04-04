export Threshold

##########################################################################################
#   Parameter
##########################################################################################
"""
    Threshold <: Parameter

The threshold is a free parameter used to classify two states as recurrent or non-recurrent.
This parameter can be optimized using the [`optimize`](@ref) function in
combination with specific __complexity measures__, e.g., **Recurrence Entropy** or
[`Disorder`](@ref).

Use:
```julia
optimize(p::Threshold, q::Entropy, N::Int, [x]; kwargs...)
optimize(p::Threshold, q::Disorder{N}, [x]; kwargs...)
```

## Arguments
- `n`: The size of the square microstate used in the optimization.
- `[x]`: The input data.

## Keyword arguments
- `ratio`: The sampling ratio. The default is `0.1`.
- `sampling`: The sampling mode. The default is [`SRandom`](@ref).
- `th_max_range`: The fraction of the maximum distance defining the upper bound of the threshold search range. The default is `0.5`.
- `th_start`: The initial value of the threshold search range. The default is `1e-6`.
- `fraction`: The interaction fraction controlling the refinement process. The default is `5`.
- `metric::Metric`: The metric used to compute recurrence.
"""
struct Threshold <: Parameter end


##########################################################################################
#   Implementation: optimize
##########################################################################################
#   - Recurrence Entropy
#.........................................................................................
function optimize(
        ::Threshold, 
        q::ComplexityMeasures.Entropy, 
        N::Int,
        x::Union{StateSpaceSet, <:AbstractGPUVector{<:SVector}},
        y::Union{StateSpaceSet, <:AbstractGPUVector{<:SVector}} = x; 
        ratio::Float64 = 0.1,
        sampling::SamplingMode = SRandom(ratio),
        th_max_range::Float64 = 0.5,
        th_start::Float64 = 1e-6,
        fraction::Int = 5,
        metric::Metric = DEFAULT_METRIC
    )

    data_x = x isa AbstractGPUVector ? x |> Vector |> StateSpaceSet : x
    data_y = y isa AbstractGPUVector ? y |> Vector |> StateSpaceSet : y

    ε = th_start
    εopt = 0.0

    if length(x) <= 1000
        εopt = maximum(pairwise(metric, x, y)) * (th_max_range - ε)
    else
        εopt = ((th_max_range - ε) / 2) * (((maximum(data_x) - minimum(data_x)))[1] * size(data_x, 2) + ((maximum(data_y) - minimum(data_y)))[1] * size(data_y, 2))
    end

    Δε = (εopt - ε) / fraction
    fmax = 0.0
    for _ ∈ 1:fraction
        for _ ∈ 1:fraction
            rmspace = RecurrenceMicrostates(ε, N; sampling = sampling)
            probs = probabilities(rmspace, x)
            f = entropy(q, probs)

            if f > fmax
                fmax = f
                εopt = ε
            end

            ε += Δε
        end

        ε = εopt - Δε
        Δε *= 2 / fraction
    end

    return εopt, fmax
end

#.........................................................................................
#   - Disorder (random sampling)
#.........................................................................................
function optimize(
        ::Threshold, 
        q::Disorder{N}, 
        x::StateSpaceSet;
        ratio::Float64 = 0.1,
        sampling::SamplingMode = SRandom(ratio),
        th_max_range::Float64 = 0.67,
        th_start::Float64 = 1e-6,
        fraction::Int = 5,
        metric::Metric = DEFAULT_METRIC
    ) where {N}

    εopt = 0.0
    ε = th_start

    if (length(x) <= 1000)
        εopt = maximum(pairwise(metric, x)) * (th_max_range - ε)
    else
        εopt = (th_max_range - ε) * ((maximum(x) - minimum(x)))[1] * size(x, 2)
    end

    Δε = (εopt - ε) / fraction
    fmax = 0.0
    for _ ∈ 1:fraction
        for _ ∈ 1:fraction
            rmspace = RecurrenceMicrostates(ε, N; sampling = sampling, metric = metric)
            partial = PartialDisorder{N}(q.labels, rmspace)
            f = complexity(partial, x)

            if f > fmax
                fmax = f
                εopt = ε
            end

            ε += Δε
        end

        ε = εopt - Δε
        Δε *= 2 / fraction
    end

    return εopt, fmax
end

##########################################################################################