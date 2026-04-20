export Disorder, WindowedDisorder

##########################################################################################
#   Quantification Measure: Determinism
#   Complexity Measure Implementation
##########################################################################################
#   Here we have four disorder definitions:
# - `Disorder`: it is the typical Disorder quantifier (https://doi.org/10.1103/1y98-x33s).
# Computed by maximizing the `PartialDisorder`.
#
# - `PartialDisorder`: it is an internal definition, which computes the entropy associated
# wit disorder for an specific threshold. It is computed using the disorder for all 
# classes, which are computed using `ClassPartialDisorder`.
#
# - `ClassPartialDisorder`: it is the entropy quantity associated with disorder for an
# specific threshold and class of microstates. We use it to compute `PartialDisorder` disorder.
#
# - `WindowedDisorder`: it is same that `Disorder`, but spliting data in windows with
# a fixed length.
##########################################################################################
"""
    Disorder{N} <: ComplexityEstimator
    Disorder(N::Int = 4; kwargs...)

An estimator of a disorder measure, introduced by [Flauzino2025Disorder](@cite), used with [complexity](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/complexity/#ComplexityMeasures.complexity).
It uses \$N \\times N\$ microstates and a specified `metric` to compute the disorder.

## Keyword arguments
- `metric::Metric`: The metric used to compute recurrence.
- `threshold_range::Int`: The number of threshold values used to estimate the disorder.

## Description
Disorder, or the disorder index via symmetry in recurrence microstates (DISREM), is based on the
implications of the disorder condition in microstates. It quantifies disorder through the
information entropy of classes of recurrence microstates, which are required to be equiprobable
within the same class according to the disorder condition.

Let \$\\sigma\$ be a permutation, and \$S_N\$ the set of all permutations of \$N\$ elements, with
\$\\sigma \\in S_N\$. Also, let \$\\mathcal{L}_\\sigma\$ be the operator that permutes the rows of a
matrix, and \$\\mathcal{T}\$ the operator that transposes a matrix. A recurrence microstate class
is defined as [Flauzino2025Disorder](@cite):
```math
\\mathscr{M}_a (\\mathbf{M}) = \\bigcup_{\\sigma_i,\\sigma_j \\in S_N}  \\{ \\mathcal{L}_{\\sigma_j}\\mathcal{T}\\mathcal{L}_{\\sigma_i}\\mathbf{M},\\quad\\mathcal{T}\\mathcal{L}_{\\sigma_j}\\mathcal{T}\\mathcal{L}_{\\sigma_i}\\mathbf{M} \\}.
```

Let \$p(\\mathbf{M})\$ be the probability of the microstate \$\\mathbf{M}\$. We renormalize this probability
with respect to its class:
```math
p^{(a)}(\\mathbf{M}) = \\frac{p(\\mathbf{M})}{\\sum_{\\mathbf{M}' \\in \\mathscr{M}_a} p(\\mathbf{M}')}.
```

The normalized information entropy associated with the probability distribution of microstates in the class
\$\\mathscr{M}_a\$ is then defined as [Flauzino2025Disorder](@cite):
```math
\\xi_a(\\varepsilon) = \\frac{1}{m_a} \\sum_{\\mathbf{M} \\in \\mathscr{M}_a} p^{(a)}(\\mathbf{M}) \\ln p^{(a)}(\\mathbf{M}),
```
where \$m_a\$ is the number of microstates in the class \$\\mathscr{M}_a\$.

By summing the entropy over all classes and normalizing by its maximum amplitude \$A\$, we obtain the total entropy
across all classes:
```math
\\xi(\\varepsilon) = \\frac{1}{A}\\sum_{a = 1}^{A} \\xi_a(\\varepsilon).
```

The disorder measure is then defined as the maximum value of \$\\xi(\\varepsilon)\$ [Flauzino2025Disorder](@cite):
```math
\\Xi = \\max_{\\varepsilon} \\xi(\\varepsilon).
```

!!! compat "Microstate size and shape"
    Disorder is only defined for square microstates, with computations available for
    \$N \\in \\{2, 3, 4, 5\\}\$ due to computational limitations. In particular,
    computations for \$N = 5\$ are computationally expensive.
"""
struct Disorder{N} <: ComplexityEstimator
    labels::Vector{Vector{Int}}
    metric::Metric
    threshold_range::Int
end

"""
    WindowedDisorder{N, W} <: ComplexityEstimator
    WindowedDisorder(W::Int, N::Int = 4; kwargs...)

This estimator is equivalent to [`Disorder`](@ref), but computes it by splitting the data into windows of
length `W`, returning a vector of measured disorder values for each window.

## Keyword arguments
- `metric::Metric`: The metric used to compute recurrence.
- `threshold_range::Int`: The number of threshold values used to estimate disorder.
- `step::Int`: The step between windows. The default is `W`.
"""
struct WindowedDisorder{W, N} <: ComplexityEstimator
    labels::Vector{Vector{Int}}
    metric::Metric
    threshold_range::Int
    win_step::Int
end

struct PartialDisorder{N} <: ComplexityEstimator
    labels::Vector{Vector{Int}}
    rmspace::RecurrenceMicrostates
end

struct ClassPartialDisorder <: ComplexityEstimator
    labels::Vector{Int}
    rmspace::RecurrenceMicrostates
end

function complexity(
        c::Disorder{N}, 
        x::Union{StateSpaceSet, Vector{<:Real}, <:AbstractGPUVector{<:SVector}}
    ) where {N}

    data = x isa Vector ? StateSpaceSet(x) : x
    data_cpu = data isa AbstractGPUVector ? data |> Vector |> StateSpaceSet : data

    #   Define the disorder range.
    th = optimize(Threshold(), c, data_cpu)[1]
    th_min = 0.7 * th
    th_max = 1.3 * th

    #   Prepare to compute.
    ξ = zeros(Float64, c.threshold_range)
    th_range = range(th_min, th_max, c.threshold_range)

    #   For GPU
    if (data isa AbstractGPUVector)
        th_range = Float32.(th_range)
    end

    #   Compute disorder for each threshold.
    for i ∈ eachindex(th_range)
        rmspace = RecurrenceMicrostates(th_range[i], N; sampling = Full(), metric = c.metric)
        partial = PartialDisorder{N}(c.labels, rmspace)
        ξ[i] = complexity(partial, data)
    end

    return maximum(ξ)
end

function complexity(
        c::WindowedDisorder{W, N},
        x::Union{StateSpaceSet, Vector{<:Real}}
    ) where {N, W}

    windowed_data = [ StateSpaceSet(x[(i + 1):(i + W)]) for i ∈ 0:c.win_step:(size(x, 1) - W) ]
    
    #   We need to define the threshold range here.
    s = ceil(Int, length(windowed_data) * 0.1)
    opt_ths = zeros(Float64, s)

    for i ∈ eachindex(s)
        idx = rand(1:length(windowed_data))
        opt_ths[i] = optimize(Threshold(), Disorder{N}(c.labels, c.metric, c.threshold_range), windowed_data[idx])[1]
    end

    μ_th = mean(opt_ths)
    σ_th = std(opt_ths)

    th_min = μ_th - 1.5 * σ_th
    th_min = th_min <= 0 ? 1e-16 : th_min
    th_max = μ_th + 1.5 * σ_th
    if th_max < th_min
        a = th_min
        th_min = th_max
        th_max = a
    end

    #   Prepare to compute.
    ξ = zeros(Float64, length(windowed_data), c.threshold_range)
    th_range = range(th_min, th_max, c.threshold_range)

    #   Finally, compute disorder for each window (note that it isn't just use "complexity(disorder, x)")
    for j ∈ eachindex(th_range)
        rmspace = RecurrenceMicrostates(th_range[j], N; sampling = Full(), metric = c.metric)
        partial = PartialDisorder{N}(c.labels, rmspace)
        
        for i ∈ eachindex(windowed_data)
            ξ[i, j] = complexity(partial, windowed_data[i])
        end
    end

    return [ maximum(ξ[i, :]) for i ∈ eachindex(windowed_data) ]
end

function complexity(
        c::WindowedDisorder{W, N},
        x::AbstractGPUVector{SVector{D, T}}
    ) where {N, W, D, T}

    #   GPU Settings
    backend = KernelAbstractions.get_backend(x)

    #   Here we have a small issue: we need to move data from GPU to CPU again, split it and
    #   then move back to the GPU... I really don't have any idea about how to create windows
    #   directly in GPU >.<
    #   And we also need to compute threshold range =/
    data = x |> Vector
    windowed_data = [ StateSpaceSet(data[(i + 1):(i + W)]) for i ∈ 0:c.win_step:(size(x, 1) - W) ]
    windowed_gpu = map(windowed_data) do w
        w_vec = w |> Vector
        gw = KernelAbstractions.allocate(backend, eltype(w_vec), size(w_vec))
        copyto!(gw, w_vec)
        gw
    end

    #   We need to define the threshold range here.
    s = ceil(Int, length(windowed_data) * 0.1) + 2
    opt_ths = zeros(Float64, s)

    for i ∈ eachindex(s)
        idx = rand(1:length(windowed_data))
        opt_ths[i] = optimize(Threshold(), Disorder{N}(c.labels, c.metric, c.threshold_range), windowed_data[idx])[1]
    end

    μ_th = mean(opt_ths)
    σ_th = std(opt_ths)

    th_min = μ_th - σ_th
    th_min = th_min <= 0 ? 1e-16 : th_min
    th_max = μ_th + σ_th
    if th_max < th_min
        a = th_min
        th_min = th_max
        th_max = a
    end

    #   Prepare to compute.
    ξ = zeros(Float64, length(windowed_gpu), c.threshold_range)
    th_range = Float32.(range(th_min, th_max, c.threshold_range))

    #   Finally, compute disorder for each window (note that it isn't just use "complexity(disorder, x)")
    for j ∈ eachindex(th_range)
        rmspace = RecurrenceMicrostates(th_range[j], N; sampling = Full(), metric = c.metric)
        partial = PartialDisorder{N}(c.labels, rmspace)
        
        for i ∈ eachindex(windowed_gpu)
            ξ[i, j] = complexity(partial, windowed_gpu[i])
        end
    end

    return [ maximum(ξ[i, :]) for i ∈ eachindex(windowed_data) ]
end

function complexity(
        c::PartialDisorder{N}, 
        x::Union{StateSpaceSet{D}, <:AbstractGPUVector{SVector{D}}}
    ) where {N, D}

    A = _norm_factor(Val(N), Val(D))
    probs = probabilities(c.rmspace, x)
    ξ = 0.0
    for i ∈ 2:(length(c.labels) - 1)
        cpartial = ClassPartialDisorder(c.labels[i], c.rmspace)
        ξ += measure(cpartial, probs)
    end

    return ξ / A
end

function complexity(
        c::ClassPartialDisorder, 
        x::Union{StateSpaceSet, <:AbstractGPUVector{<:SVector}}
    )
    probs = probabilities(c.rmspace, x)
    return measure(c, probs)
end

# -- Constructors
Disorder(N::Int = 4; metric::Metric = DEFAULT_METRIC, threshold_range::Int =  40) = Disorder{N}(compute_labels(N), metric, threshold_range)
WindowedDisorder(W::Int, N::Int = 4; metric::Metric = DEFAULT_METRIC, threshold_range::Int = 40, step::Int = W) = WindowedDisorder{W,N}(compute_labels(N), metric, threshold_range, step)
PartialDisorder(rexpr::RecurrenceExpression, N::Int = 4) = PartialDisorder{N}(compute_labels(N), RecurrenceMicrostates(rexpr, N; sampling = Full()))
ClassPartialDisorder(rexpr::RecurrenceExpression, c::Int, N::Int = 4) = ClassPartialDisorder(compute_labels(N)[c], RecurrenceMicrostates(rexpr, N; sampling = Full()))

##########################################################################################
#   Internal: measure from probabilities
##########################################################################################
# This is an internal function which estimates the determinism from a recurrence microstate
# outcome space, using a given probability distribution that was computed from this
# outcome space.

# This function works for [`DiagonalMicrostate`](@ref) with length 3, 
# or \$3 \\times 3\$ [`RectMicrostate`](@ref). Any other input will returns an error.
##########################################################################################
function measure(c::ClassPartialDisorder, probs::Probabilities)

    norm_factor = 0.0
    @inbounds @simd for i ∈ c.labels
        norm_factor += probs[i]
    end

    if (norm_factor <= 0.0)
        return 0.0
    end

    ξ = 0.0
    @inbounds @simd for i ∈ c.labels
        p = probs[i] / norm_factor
        ξ += p * log(p + eps())
    end

    ξ *= -1 / log(length(c.labels))
    return ξ
end

##########################################################################################
#   Internal: Aux. functions
##########################################################################################
_norm_factor(::Val{2}, ::Val{D}) where D = 4
_norm_factor(::Val{3}, ::Val{D}) where D = D > 1 ? 24 : 23
_norm_factor(::Val{4}, ::Val{D}) where D = D > 1 ? 190 : 145
_norm_factor(::Val{5}, ::Val{D}) where D = D > 1 ? throw(ArgumentError("Disorder not implemented using N = 5 for data with more than one dimension.")) : 1173

function compute_labels(N::Int; S = collect(permutations(1:N)))
    shape = RectMicrostate(N)

    row_permutation = PermuteRows(shape)
    col_permutation = PermuteColumns(shape; S = S)
    transposition = Transpose(shape)

    identified = Set{Int}()
    labels = Vector{Vector{Int}}(undef, 0)

    for i ∈ 1:(2^(N * N))

        #   Verify if `i` was identified or not.
        if i ∈ identified
            continue
        end

        #   Create a new class.
        push!(labels, Vector{Int}(undef, 1))
        push!(identified, i)
        labels[end][1] = i

        #   Compute the permutations of `i`
        for col in eachindex(col_permutation.Q)
            for row in eachindex(S)

                #   1. Permute rows
                microstate = operate(row_permutation, i, S[row])
                #   2. Permute columns
                microstate = operate(col_permutation, microstate, col)

                #   Security verify
                if microstate ∈ identified
                    continue
                end

                #   Ad the new label to the class
                push!(identified, microstate)
                push!(labels[end], microstate)
            end
        end

        #   Compute transposes
        for label in copy(labels[end])
            
            #   Transpose the label
            microstate = operate(transposition, label)

            #   Security verify
            if microstate ∈ identified
                continue
            end

            #   Ad the new label to the class
            push!(identified, microstate)
            push!(labels[end], microstate)
        end
    end

    return labels
end

##########################################################################################

ComplexityMeasures.relevant_fieldnames(::Disorder) = [:metric, :threshold_range]
ComplexityMeasures.relevant_fieldnames(::WindowedDisorder) = [:metric, :threshold_range, :win_step]

##########################################################################################