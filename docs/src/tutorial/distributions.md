#   Distributions

This section introduces the computation of Recurrence Microstates Analysis (RMA) distributions
using **RecurrenceMicrostatesAnalysis.jl**.

We begin with a [Quick start with RecurrenceMicrostatesAnalysis.jl](@ref), which demonstrates a
simple application example. Next, we present [A brief review](@ref) of Recurrence Plots (RP)
and RMA. Finally, we explain the [`distribution`](@ref) function in
[Computing RMA distributions](@ref), including the role of [Histograms](@ref).

##  Computing RMA distributions

The computation of RMA distributions is the core functionality of
RecurrenceMicrostatesAnalysis.jl. All other tools in the package rely on these
distributions as their primary source of information.

RMA distributions are computed using the [`distribution`](@ref) function:
```@docs
distribution
```

A commonly used convenience interface is:
```julia
distribution([x], ε::Float, n::Int; kwargs...)
```

## Spatial data

The package also provides experimental support for spatial data, following *"Generalised Recurrence Plot Analysis for Spatial Data"* [Marwan2007Spatial](@cite).
In this context, input data are provided as `AbstractArray`s:
```math
    \vec{x}_{\vec i} \in \mathbb{R}^m,\quad \vec{i} \in \mathbb{Z}^d
```

For example:
```@example quick_example
spatialdata = rand(Uniform(0, 1), (2, 50, 50))
```
Due to the high dimensionality of spatial recurrence plots, direct visualization is often
infeasible. RMA distributions provide a compact alternative by sampling microstates directly
from the data.

**Examples:**
- Full $2 \times d$ microstates:
```@example quick_example
distribution(spatialdata, Rect(Standard(0.27), (2, 2, 2, 2)))
```
```@example quick_example
spatialdata_1 = rand(Uniform(0, 1), (2, 50, 50))
spatialdata_2 = rand(Uniform(0, 1), (2, 25, 25))
distribution(spatialdata_1, spatialdata_2, Rect(Standard(0.27), (2, 2, 2, 2)))
```

- Projected microstates:
```@example quick_example
distribution(spatialdata, Rect(Standard(0.27), (2, 1, 2, 1)))
```
```@example quick_example
distribution(spatialdata_1, spatialdata_2, Rect(Standard(0.27), (2, 1, 2, 1)))
```
