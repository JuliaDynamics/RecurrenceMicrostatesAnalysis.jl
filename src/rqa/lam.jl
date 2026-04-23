export RecurrenceLaminarity

##########################################################################################
#   Quantification Measure: Laminarity
#   Complexity Measure Implementation
##########################################################################################
"""
    RecurrenceLaminarity <: ComplexityEstimator
    RecurrenceLaminarity(ε::Float64; kwargs...)

An estimator of recurrence laminarity, used with [complexity](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/complexity/#ComplexityMeasures.complexity).
Laminarity is estimated for a threshold `ε`.

## Keyword arguments
- `metric::Metric`: The metric used to compute recurrence.
- `ratio`: The sampling ratio. The default is `0.1`.
- `sampling`: The sampling mode. The default is [`SRandom`](@ref).

## Description

Recurrence laminarity (LAM) is defined as [Webber2015Recurrence](@cite)
```math
LAM = \\frac{\\sum_{l=l_{min}}^{K} l H_V(l)}{\\sum_{i,j=1}^{K} r_{(i,j)}},
```
where \$H_V(l)\$ is the histogram of vertical line lengths:
```math
H_V(l) = \\sum_{i,j=1}^{K} (1 - r_{i, j-1})(1 - r_{i,j+l})\\prod_{k=0}^{l-1} r_{i,j+k}.
```
By inverting the laminarity expression, we can rewrite it as [daCruz2025RQAMeasures](@cite)
```math
LAM = 1 - \\frac{1}{K^2 \\sum_{i,j=1}^{K} r_{(i,j)}} \\sum_{l=1}^{l_{min} - 1} l H_V(l).
```

An approximate value for LAM can be estimated using recurrence microstates, as introduced by
da Cruz et al. [daCruz2025RQAMeasures](@cite). From an input dataset `x`, we estimate a
recurrence microstate distribution \$\\vec{p}\$. This distribution must be defined over
square microstates of size \$3 \\times 3\$. Here, we use the relation:
```math
\\frac{H_V(l)}{(K-l-1)^2} = \\vec{v}^{(l)} \\cdot \\mathcal{R}^{(l + 2)}\\vec{p}^{(l + 2)}.
```

For the commonly used case \$l_{min} = 2\$, this leads to the approximation
```math
LAM \\approx  1 - \\frac{\\vec{v}^{(1)}\\cdot\\mathcal{R}^{(3)}\\vec{p}^{(3)}}{\\sum_{i,j=1}^{K} r_{(i,j)}}.
```

The correlation term \$\\vec{v}^{(1)} \\cdot \\mathcal{R}^{(3)} \\vec{p}^{(3)}\$ can be 
simplified by explicitly identifying the microstates selected by \$\\vec{v}^{(1)}\$. This corresponds
to selecting only microstates of the form:
```math
\\begin{pmatrix}
0 & 1 & 0 \\\\
\\xi & \\xi & \\xi \\\\
\\xi & \\xi & \\xi
\\end{pmatrix},
```
where \$\\xi\$ denotes an unconstrained entry. There are 64 microstates with this structure among
the 512 possible \$3 \\times 3\$ microstates. Defining the class \$C_V\$ as the set of microstates 
with this structure, LAM can be estimated as:
```math
LAM \\approx 1 - \\frac{\\sum_{i\\in C_V} p_i^{(3)}}{\\sum_{i,j=1}^{K} r_{(i,j)}}.
```

The implementation used by [complexity](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/complexity/#ComplexityMeasures.complexity)
is an optimized version of this process using \$1 \\times 3\$ [`RectMicrostate`](@ref) [Ferreira2025RMALib](@cite).
Since this microstate shape is symmetric with respect to the desired information, it is not necessary to account for
\$\\xi\$ values as in the square microstate case. Thus, laminarity can be estimated as
```math
LAM \\approx 1 - \\frac{p_3^{(3)}}{\\sum_{i,j=1}^{K} r_{(i,j)}},
```
where \$p_3^{(3)}\$ is the probability of observing the microstate \$0~1~0\$.

!!! note "Performance"
    Although estimating LAM using RMA is faster than typical RQA computation,
    the precision depends on the time series length. Therefore, for small time series,
    i.e., \$K \\leq 1000\$, we strongly recommend using standard RQA with
    [RecurrenceAnalysis.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/recurrenceanalysis/stable/quantification/#RecurrenceAnalysis.laminarity).
"""
struct RecurrenceLaminarity{M, SM} <: ComplexityEstimator 
    ε::Real
    metric::M
    sampling::SM
end

function complexity(
        c::RecurrenceLaminarity, 
        x::Union{StateSpaceSet, Vector{<:Real}, <:AbstractGPUVector{<:SVector}}, 
        y::Union{StateSpaceSet, Vector{<:Real}, <:AbstractGPUVector{<:SVector}} = x;
    )

    rmspace = RecurrenceMicrostates(c.ε, RectMicrostate(1, 3); metric = c.metric, sampling = c.sampling)
    probs = probabilities(rmspace, x, y)
    return measure(c, rmspace, probs)
end

# -- Constructors
RecurrenceLaminarity(ε::Real; metric::M = DEFAULT_METRIC, ratio::Float64 = 0.1, sampling::SM = SRandom(ratio)) where {M,SM} = RecurrenceLaminarity{M,SM}(ε, metric, sampling)

##########################################################################################
#   Internal: measure from probabilities
##########################################################################################
# This is an internal function which estimates the determinism from a recurrence microstate
# outcome space, using a given probability distribution that was computed from this
# outcome space.

# This function works for [`DiagonalMicrostate`](@ref) with length 3, 
# or \$3 \\times 3\$ [`RectMicrostate`](@ref). Any other input will returns an error.
##########################################################################################
function measure(c::RecurrenceLaminarity, rmspace::RecurrenceMicrostates, probs::Probabilities)
    rrc = RecurrenceRate(c.ε; metric = c.metric, sampling = c.sampling)
    if (rmspace.shape isa Rect2Microstate{3, 3, 2} && length(probs) == 512)
        rr = measure(rrc, probs)

        values = zeros(Int, 64)
        v_idx = 1

        for a1 in 0:1, a2 in 0:1, a3 in 0:1, a4 in 0:1, a5 in 0:1, a6 in 0:1
            I_1 = 2 + 8 * a1 + 16 * a2 + 32 * a3 + 64 * a4 + 128 * a5 + 256 * a6
            values[v_idx] = I_1 + 1
            v_idx += 1
        end

        pl = 0.0
        for i in values
            pl += probs[i]
        end

        return 1 - ((1/rr) * pl)

   elseif (rmspace.shape isa Rect2Microstate{1, 3} && length(probs) == 8)
        rr = measure(rrc, probs)
        return 1 - ((1/rr) * probs[3])
    else
        msg = "Laminarity must be computed using square or line microstates with n = 3."
        throw(ArgumentError(msg))
    end
end

##########################################################################################