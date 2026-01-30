#   RecurrenceMicrostatesAnalysis.jl

```@docs
RecurrenceMicrostatesAnalysis
```

!!! info "Citation and credit"
    If you find this package useful, please consider giving it a star on GitHub and don't forget to cite [our work](https://doi.org/10.1063/5.0293708). 😉


##  About the documentation
The documentation of **RecurrenceMicrostatesAnalysis.jl** is designed to explain how to use the package while also introducing the theoretical background of the RMA framework. The bibliography used throughout the documentation is listed in the [References](@ref) section; **please remember to cite the appropriate works if you use them**.

This welcome section begins with an introduction to the [Input data for RecurrenceMicrostatesAnalysis.jl](@ref). Understanding the data types used by the package and their intended purposes is essential before proceeding with the rest of the documentation. We also describe the [Output data from RecurrenceMicrostatesAnalysis.jl](@ref), explaining the type of data returned when computing recurrence microstate distributions.

The **Tutorial** section explains how to use the package in practice. It starts with a brief introduction to RMA and demonstrates how to compute [Distributions](@ref) using **RecurrenceMicrostatesAnalysis.jl**. Next, we show how to estimate RQA [Quantifiers](@ref) using RMA and discuss quantifiers defined specifically for RMA. This material constitutes the *basic level* of the documentation and is sufficient to learn how to effectively use this package.

For users interested in more advanced topics, the [Recurrence Functions](@ref) section discusses different ways of computing recurrence between two states, while the [Shapes and Sampling](@ref) section explains motif shapes used to extract specific information from a Recurrence Plot.

We also provide a pipeline for [GPU](@ref) computations, which we recommend reading if you plan to use the GPU backend.

The documentation includes applied examples, such as:
- [RMA with Machine Learning](@ref)

Finally, developers interested in contributing to RecurrenceMicrostatesAnalysis.jl are encouraged to read the [RecurrenceMicrostatesAnalysis.jl for Devs](@ref) section.

##  Input data for RecurrenceMicrostatesAnalysis.jl
**RecurrenceMicrostatesAnalysis.jl** accepts three types of input, each associated with a different backend:

- [`StateSpaceSet`](@ref) — used for multivariate time series, datasets, or state-space representations. This type is employed when working with Recurrence Plots (RP) or Cross-Recurrence Plots (CRP). For RP and CRP analyses, we strongly recommend using this data type, as the backend is optimized for this context.

- `AbstractArray{<: Real}` — used for spatial data, enabling RMA to be applied within the generalized framework of Spatial Recurrence Plots (SRP) [Marwan2007Spatial](@cite). Although a `Matrix` can be used as a substitute for a [`StateSpaceSet`](@ref), this is **not recommended**, since the `AbstractArray` backend is heavier and incompatible with some features.

- `AbstractGPUVector` — used for time series analysis with the GPU backend. A better explanation is provided in the [GPU](@ref) and [Computing RMA distributions](@ref) sections.

!!! warning
    RMA with SRP is an open research field. We include this functionality in the package for exploratory purposes, but the method is not
    yet mature enough for production use. Nevertheless, feel free to experiment with it in your research. 😃

```@docs
StateSpaceSet
```

##  Output data from RecurrenceMicrostatesAnalysis.jl
When computing the RMA distribution, RecurrenceMicrostatesAnalysis.jl returns a [`Probabilities`](@ref) structure. This type is
provided by [ComplexityMeasures.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/), allowing this package
to interoperate naturally with its tools and workflows.

```@docs
Probabilities
Counts
```