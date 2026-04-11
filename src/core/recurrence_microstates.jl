export RecurrenceMicrostates

##########################################################################################
#   Recurrence Microstate
##########################################################################################
"""
    RecurrenceMicrostates <: CountBasedOutcomeSpace

It defines an `OutcomeSpace` from **ComplexityMeasures.jl**, representing a recurrence microstate
outcome space.

## Description

Let \$\\mathscr{X} = (\\vec{x}_i)_{i = 1}^{K}\$ be a sequence of data elements. From \$\\mathscr{X}\$,
we can extract subsequences \$\\mathbf{X}_{(p)}\$ of length \$N\$ such that
\$\\mathbf{X}_{(p)} = (\\vec{x}_i)_{i=p+1}^{p+N}\$ for \$0 \\leq p \\leq K - N\$.

Let \$r_{(i,j)}\$ be a recurrence function that takes two elements, \$\\vec{x}_i\$ and \$\\vec{x}_j\$, and returns whether they
are recurrent. For example, the threshold-based recurrence is given by
\$r_{(i,j)} = \\Theta(\\varepsilon - \\|\\vec{x}_i - \\vec{x}_j\\|)\$,
where \$\\Theta\$ is the Heaviside function, \$\\|\\cdot\\|\$ is an appropriate `metric` for the data, and \$\\varepsilon\$ is
the recurrence threshold. In this representation, a recurrence is denoted by **1** and a non-recurrence by **0**.

For two subsequences belonging to \$\\mathscr{X}\$, we can generate, using the recurrence function, a
matrix \$\\mathbf{M} \\equiv \\mathbf{R}(\\mathbf{X}_{(p)}, \\mathbf{X}_{(q)})\$:
```math
\\mathbf{R}(\\mathbf{X}_{(p)}, \\mathbf{X}_{(q)}) \\coloneqq \\begin{bmatrix}
r_{p+1,q+1} & \\dots & r_{p+1,q+N} \\\\
\\vdots & \\ddots & \\vdots \\\\
r_{p+N,q+1} & \\dots & r_{p+N,q+N}
\\end{bmatrix}.
```
This matrix is known as a recurrence microstate [Corso2018Entropy](@cite) of length \$N\$ when \$N \\ll K\$, and represents
a local portion of the recurrence plot \$\\mathbf{R}(\\mathscr{X}, \\mathscr{X})\$.

## Implementation
To define a recurrence microstate, three components are required:
1. The recurrence function, \$r_{(i,j)}\$, used to compute recurrences (see [`RecurrenceExpression`](@ref)).
2. The microstate shape and its size (see [`MicrostateShape`](@ref)).
3. The method used to extract microstates from \$\\mathscr{X}\$ (see [`SamplingMode`](@ref)).

## Constructors
```julia
RecurrenceMicrostates(expr::RecurrenceExpression, shape::MicrostateShape; kwargs...)
RecurrenceMicrostates(expr::RecurrenceExpression, N::Int; kwargs...) # It uses square microstates.
```

- Using [`ThresholdRecurrence`](@ref):
```julia
RecurrenceMicrostates(ε::Real, N::Int; kwargs...)
RecurrenceMicrostates(ε::Real, shape::MicrostateShape; kwargs...)
```

- Using [`CorridorRecurrence`](@ref):
```julia
RecurrenceMicrostates(ε_min::Real, ε_max::Real, N::Int; kwargs...)
RecurrenceMicrostates(ε_min::Real, ε_max::Real, shape::MicrostateShape; kwargs...)
```

## Spatial generalization
We also implement a spatial generalization of recurrence microstate analysis based on [Marwan2007Spatial](@cite).
It operates similarly to standard RMA, but retrieves microstates from a recurrence tensor constructed
from the data (without explicitly constructing the full tensor). **This is an experimental feature included
as an exploratory addition to the package; there is currently no established literature describing how to use
this generalization.** If you are curious, feel free to experiment with it 🙂

In this setting, the input data is no longer a sequence of elements, but a spatial structure with \$d\$ dimensions.
Each element of this structure can be either a `Real` or a `Vector` containing features associated with the corresponding
position. Following [Marwan2007Spatial](@cite), the resulting recurrence space has \$2 \\times d\$ dimensions. For example,
for an image we have \$d = 2\$, so the recurrence space has 4 dimensions.

The recurrence expression is reformulated to operate on vector indices: \$r_{(\\vec{i}, \\vec{j})}\$. For example, the threshold-based
recurrence is given by
\$r_{(\\vec{i}, \\vec{j})} = \\Theta(\\varepsilon - \\|\\mathbf{x}_{\\vec{i}} - \\mathbf{x}_{\\vec{j}}\\|)\$,
where the coordinate vectors are defined as:
```math
\\vec{i} = \\sum_{q = 1}^{d} i_q \\hat{e}_q,
```
where \$\\hat{e}_q\$ is the unit vector representing the corresponding dimension \$q\$.

Using this formulation, recurrence microstates are defined as before. However, since the recurrence plot is now a tensor,
microstates may have hypergeometric shapes, such as hypercubes or hyperrectangles, or correspond to projections onto
lower-dimensional subspaces. This provides greater flexibility in capturing structure, but also increases the complexity
of using the method.

Moreover, since RQA is defined differently for spatial recurrence plots, the implementations in
**RecurrenceMicrostatesAnalysis.jl** are generally not compatible with this generalization, except for
[`RecurrenceRate`](@ref) and **Entropy**, which can be estimated from any recurrence microstate distribution.

!!! compat "Shape and sampling compatibility"
    Some shapes and sampling modes are not compatible with the spatial generalization, e.g. [`TriangleMicrostate`](@ref) and [`Full`](@ref).

### Spatial constructors
These constructors use a hypergeometric version of [`RectMicrostate`](@ref), defined by an `NTuple` called `structure`.

```julia
RecurrenceMicrostates(expr::RecurrenceExpression, structure::NTuple; kwargs...)
RecurrenceMicrostates(ε::Real, structure::NTuple; kwargs...)
RecurrenceMicrostates(ε_min::Real, ε_max::Real, structure::NTuple; kwargs...)
```

## Keyword arguments
- `metric::Metric`: The metric used to compute recurrence.
- `ratio`: The sampling ratio. The default is `0.1`.
- `sampling`: The sampling mode. The default is [`SRandom`](@ref).

!!! warning "Memory"
    Note that the number of possible microstates is \$2^\\sigma\$, where
    \$\\sigma\$ is the number of recurrence entries in the microstate structure.
    This means that as the number of entries increases, the memory required to
    store the full distribution quickly becomes impractical. For example, a square
    \$6 \\times 6\$ microstate has 36 entries, resulting in \$2^{36} = 68,719,476,736\$
    possible microstates. As another example, a 4-dimensional hypercubic microstate
    with side length 3 has \$3^4 = 81\$ entries, leading to \$2^{81} \\approx 2.42 \\times 10^{24}\$
    possible microstates. Clearly, allocating memory for such distributions is not feasible.

"""
struct RecurrenceMicrostates{MS <: MicrostateShape, RE <: RecurrenceExpression, SM <: SamplingMode} <: ComplexityMeasures.CountBasedOutcomeSpace
    shape::MS
    expr::RE
    sampling::SM
end

##########################################################################################
#   Recurrence Microstate: Convenience constructors
##########################################################################################
function RecurrenceMicrostates(expr::RecurrenceExpression, shape::MicrostateShape; ratio::Real = 0.05, sampling::SamplingMode = SRandom(ratio))
    return RecurrenceMicrostates(shape, expr, sampling)
end

function RecurrenceMicrostates(expr::RecurrenceExpression, N::Int; ratio::Real = 0.05, sampling::SamplingMode = SRandom(ratio))
    shape = RectMicrostate(N)
    return RecurrenceMicrostates(shape, expr, sampling)
end
##########################################################################################
function RecurrenceMicrostates(ε::Real, N::Int; ratio::Real = 0.05, sampling::SamplingMode = SRandom(ratio), metric::Metric = DEFAULT_METRIC)
    shape = RectMicrostate(N)
    expr = ThresholdRecurrence(ε; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling)
end

function RecurrenceMicrostates(ε::Real, shape::MicrostateShape; ratio::Real = 0.05, sampling::SamplingMode = SRandom(ratio), metric::Metric = DEFAULT_METRIC)
    expr = ThresholdRecurrence(ε; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling)
end

##########################################################################################
function RecurrenceMicrostates(ε_min::Real, ε_max::Real, N::Int; ratio::Real = 0.05, sampling::SamplingMode = SRandom(ratio), metric::Metric = DEFAULT_METRIC)
    shape = RectMicrostate(N)
    expr = CorridorRecurrence(ε_min, ε_max; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling)
end

function RecurrenceMicrostates(ε_min::Real, ε_max::Real, shape::MicrostateShape; ratio::Real = 0.05, sampling::SamplingMode = SRandom(ratio), metric::Metric = DEFAULT_METRIC)
    expr = CorridorRecurrence(ε_min, ε_max; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling)
end
##########################################################################################
function RecurrenceMicrostates(expr::RecurrenceExpression, structure::NTuple; ratio::Real = 0.05, sampling::SamplingMode = SRandom(ratio))
    shape = RectMicrostate(structure)
    return RecurrenceMicrostates(shape, expr, sampling)
end

##########################################################################################
function RecurrenceMicrostates(ε::Real, structure::NTuple; ratio::Real = 0.05, sampling::SamplingMode = SRandom(ratio), metric::Metric = DEFAULT_METRIC)
    shape = RectMicrostate(structure)
    expr = ThresholdRecurrence(ε; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling)
end

##########################################################################################
function RecurrenceMicrostates(ε_min::Real, ε_max::Real, structure::NTuple; ratio::Real = 0.05, sampling::SamplingMode = SRandom(ratio), metric::Metric = DEFAULT_METRIC)
    shape = RectMicrostate(structure)
    expr = CorridorRecurrence(ε_min, ε_max; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling)
end


