export RecurrenceDeterminism

##########################################################################################
#   Quantification Measure: Determinism
#   Complexity Measure Implementation
##########################################################################################
"""
    RecurrenceDeterminism <: ComplexityEstimator
    RecurrenceDeterminism(ε::Float64; kwargs...)

An estimator of recurrence determinism, used with [complexity](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/complexity/#ComplexityMeasures.complexity).
Determinism is estimated for a threshold `ε`.

## Keyword arguments
- `metric::Metric`: The metric used to compute recurrence.
- `ratio`: The sampling ratio. The default is `0.1`.
- `sampling`: The sampling mode. The default is [`SRandom`](@ref).

## Description

Recurrence determinism (DET) is defined as [Webber2015Recurrence](@cite)
```math
DET = \\frac{\\sum_{l=l_{min}}^{K} l H_D(l)}{\\sum_{i,j=1}^{K} r_{i,j}},
```
where \$H_D(l)\$ is the histogram of diagonal line lengths:
```math
H_D(l) = \\sum_{i,j=1}^{K} (1 - r_{i-1, j-1})(1 - r_{i+l,j+l})\\prod_{k=0}^{l-1} r_{i+k,j+k}.
```
By inverting the determinism expression, we can rewrite it as [daCruz2025RQAMeasures](@cite)
```math
DET = 1 - \\frac{1}{K^2 \\sum_{i,j=1}^{K} r_{i,j}} \\sum_{l=1}^{l_{min} - 1} l H_D(l).
```

An approximate value for DET can be estimated using recurrence microstates, as introduced by
da Cruz et al. [daCruz2025RQAMeasures](@cite). From an input dataset `x`, we estimate a
recurrence microstate distribution \$\\vec{p}\$. This distribution must be defined over
square microstates of size \$3 \\times 3\$. Here, we use the relation:
```math
\\frac{H_D(l)}{(K-l-1)^2} = \\vec{d}^{(l)} \\cdot \\mathcal{R}^{(l + 2)}\\vec{p}^{(l + 2)}.
```

For the commonly used case \$l_{min} = 2\$, this leads to the approximation
```math
DET \\approx  1 - \\frac{\\vec{d}^{(1)}\\cdot\\mathcal{R}^{(3)}\\vec{p}^{(3)}}{\\sum_{i,j=1}^{K} r_{i,j}}.
```

The correlation term \$\\vec{d}^{(1)} \\cdot \\mathcal{R}^{(3)} \\vec{p}^{(3)}\$ can be 
simplified by explicitly identifying the microstates selected by \$\\vec{d}^{(1)}\$. This corresponds
to selecting only microstates of the form:
```math
\\begin{pmatrix}
\\xi & \\xi & 0 \\\\
\\xi & 1 & \\xi \\\\
0 & \\xi & \\xi
\\end{pmatrix},
```
where \$\\xi\$ denotes an unconstrained entry. There are 64 microstates with this structure among
the 512 possible \$3 \\times 3\$ microstates. Defining the class \$C_D\$ as the set of microstates 
with this structure, DET can be estimated as:
```math
DET \\approx 1 - \\frac{\\sum_{i\\in C_D} p_i^{(3)}}{\\sum_{i,j=1}^{K} r_{i,j}}.
```

The implementation used by [complexity](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/complexity/#ComplexityMeasures.complexity)
is an optimized version of this process using [`DiagonalMicrostate`](@ref) [Ferreira2025RMALib](@cite).
Since this microstate shape is symmetric with respect to the desired information, it is not necessary to account for
\$\\xi\$ values as in the square microstate case. Thus, determinism can be estimated as
```math
DET \\approx 1 - \\frac{p_3^{(3)}}{\\sum_{i,j=1}^{K} r_{i,j}},
```
where \$p_3^{(3)}\$ is the probability of observing the microstate \$0~1~0\$.

!!! note "Performance"
    Although estimating DET using RMA is faster than typical RQA computation,
    the precision depends on the time series length. Therefore, for small time series,
    i.e., \$K \\leq 1000\$, we strongly recommend using standard RQA with
    [RecurrenceAnalysis.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/recurrenceanalysis/stable/quantification/#RecurrenceAnalysis.determinism).
"""
struct RecurrenceDeterminism <: ComplexityEstimator 
    ε::Float64
    metric::Metric
    sampling::SamplingMode
end

function complexity(
        c::RecurrenceDeterminism, 
        x::Union{StateSpaceSet, Vector{<:Real}, <:AbstractGPUVector{SVector}}, 
        y::Union{StateSpaceSet, Vector{<:Real}, <:AbstractGPUVector{SVector}} = x;
    )

    rmspace = RecurrenceMicrostates(c.ε, DiagonalMicrostate(3); metric = c.metric, sampling = c.sampling)
    probs = probabilities(rmspace, x, y)
    return measure(c, rmspace, probs)
end

# -- Constructors
RecurrenceDeterminism(ε; metric::Metric = DEFAULT_METRIC, ratio::Float64 = 0.1, sampling::SamplingMode = SRandom(ratio)) = RecurrenceDeterminism(ε, metric, sampling)

##########################################################################################
#   Internal: measure from probabilities
##########################################################################################
# This is an internal function which estimates the determinism from a recurrence microstate
# outcome space, using a given probability distribution that was computed from this
# outcome space.

# This function works for [`DiagonalMicrostate`](@ref) with length 3, 
# or \$3 \\times 3\$ [`RectMicrostate`](@ref). Any other input will returns an error.
##########################################################################################
function measure(c::RecurrenceDeterminism, rmspace::RecurrenceMicrostates, probs::Probabilities)
    rrc = RecurrenceRate(c.ε; metric = c.metric, sampling = c.sampling)
    if (rmspace.shape isa Rect2Microstate{3, 3, 2} && length(probs) == 512)
        rr = measure(rrc, probs)
        values = zeros(Int, 64)
        v_idx = 1

        for a1 ∈ 0:1, a2 ∈ 0:1, a3 ∈ 0:1, a4 ∈ 0:1, a5 ∈ 0:1, a6 ∈ 0:1
            I_1 = 2 * a1 + 4 * a2 + 8 * a3 + 16 + 32 * a4 + 64 * a5 + 128 * a6
            values[v_idx] = I_1 + 1
            v_idx += 1
        end

        pl = 0.0
        for i ∈ values
            pl += probs[i]
        end

        return 1 - ((1/rr) * pl)

    elseif (rmspace.shape isa DiagonalMicrostate{3, 2} && length(probs) == 8)
        rr = measure(rrc, probs)
        return 1 - ((1/rr) * probs[3])
    else
        msg = "Determinism must be computed using square or diagonal microstates with n = 3."
        throw(ArgumentError(msg))
    end
end

##########################################################################################