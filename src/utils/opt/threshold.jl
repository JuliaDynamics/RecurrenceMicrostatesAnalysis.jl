export Threshold

##########################################################################################
#   Parameter
##########################################################################################
"""
    Threshold <: Parameter

Threshold parameter used to classify two states as recurrent or non-recurrent.

The `Threshold` parameter can be optimized using the [`optimize`](@ref) function in
combination with specific [`QuantificationMeasure`](@ref)s:
```julia
optimize(::Threshold, qm::RecurrenceEntropy, [x], n::int; kwargs...)
optimize(::Threshold, qm::Disorder{N}, [x]; kwargs...)
```

!!! compat
    Threshold optimization using RMA is currently supported only for the
    [`RecurrenceEntropy`](@ref) and [`Disorder`](@ref) quantification measures.

#   Arguments
- `qm`: A [`QuantificationMeasure`](@ref) used to determine the optimal threshold. Supported measures are [`RecurrenceEntropy`](@ref) and [`Disorder`](@ref).
- `[x]`: Input data used to estimate the optimal threshold.
- `n`: Size of the square microstate used in the optimization.

#   Returns
A `Tuple{Float64, Float64}`, where:
- the first element is the optimal threshold value, and
- the second element is the value of the corresponding [`QuantificationMeasure`](@ref) at the optimum.

#   Keyword Arguments
- `rate`: Sampling rate. Default is `0.05`.
- `sampling`: Sampling mode. Default is [`SRandom`](@ref).
- `th_max_range`: Fraction of the maximum distance defining the upper bound of the threshold search range. Default is `0.5`.
- `th_start`: Initial value of the threshold search range. Default is `1e-6`.
- `fraction`: Interaction fraction controlling the refinement process. Default is `5`.

#   Example
```julia
using Distributions, RecurrenceMicrostatesAnalysis
data = StateSpaceSet(rand(Uniform(0, 1), 1000))
th, s = optimize(Threshold(), RecurrenceEntropy(), data, 3)
```
"""
struct Threshold <: Parameter end

##########################################################################################
#   Implementation: optimize
##########################################################################################
#   - Recurrence Entropy
#.........................................................................................
function optimize(
        ::Threshold, 
        qm::RecurrenceEntropy, 
        x, 
        N::Int; 
        rate::Float64 = 0.05,
        sampling::SamplingMode = SRandom(rate),
        th_max_range::Float64 = 0.5,
        th_start::Float64 = 1e-6,
        fraction::Int = 5,
    )

    ε = th_start
    εopt = 0.0

    if length(x) <= 1000
        εopt = maximum(pairwise(Euclidean(), x)) * (th_max_range - ε)
    else
        εopt = ((maximum(x) - minimum(x)))[1] * size(x, 2)
    end

    Δε = (εopt - ε) / fraction
    fmax = 0.0
    for _ ∈ 1:fraction
        for _ ∈ 1:fraction
            rmspace = RecurrenceMicrostates(ε, N; sampling = sampling)
            probs = probabilities(rmspace, x)
            f = measure(qm, probs)

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
        qm::Disorder{N}, 
        x;
        rate::Float64 = 0.05,
        sampling::SamplingMode = SRandom(rate),
        th_max_range::Float64 = 0.5,
        th_start::Float64 = 1e-6,
        fraction::Int = 5,
    ) where {N}

    ε = th_start
    εopt = 0.0

    if length(x) <= 1000
        εopt = maximum(pairwise(Euclidean(), x)) * (th_max_range - ε)
    else
        εopt = ((maximum(x) - minimum(x)))[1] * size(x, 2)
    end

    A = get_disorder_norm_factor(qm, x)
    Δε = (εopt - ε) / fraction
    fmax = 0.0
    for _ ∈ 1:fraction
        for _ ∈ 1:fraction
            rmspace = RecurrenceMicrostates(ε, N; sampling = sampling)
            probs = probabilities(rmspace, x)
            f = measure(qm, probs, A)

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