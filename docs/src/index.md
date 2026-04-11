#   RecurrenceMicrostatesAnalysis.jl

!!! info "Citation and credit"
    If you find this package useful, please consider giving it a star on GitHub and don't forget to cite [our work](https://doi.org/10.1063/5.0293708). 😉

##  Latest news
- Integration with **DynamicalSystems.jl** and **ComplexityMeasures.jl**.
- See the **CHANGELOG.md** (at the GitHub repo) for more!

##  Getting started
Start by reviewing [Input data for RecurrenceMicrostatesAnalysis.jl](@ref) and
[Output data from RecurrenceMicrostatesAnalysis.jl](@ref). Then, you can explore the
[Tutorial for RecurrenceMicrostatesAnalysis.jl](@ref) for a basic introduction to using the package.
You can also consult individual functions in the [API](@ref), and find applied examples in the
dedicated [Examples](@ref) section.

We also provide a section [RecurrenceMicrostatesAnalysis.jl for devs](@ref) for those interested
in developing new methods for **RecurrenceMicrostatesAnalysis.jl**, such as new microstate shapes,
sampling modes, recurrence functions, or complexity estimators.

### Input data for RecurrenceMicrostatesAnalysis.jl
**RecurrenceMicrostatesAnalysis.jl** accepts three types of input, each associated with a different backend:
- [`StateSpaceSet`] or `Vector{<:Real}`: used for multivariate time series, datasets, or state-space representations.
- `AbstractArray{<:Real}`: used for spatial data, enabling RMA to be applied within the generalized framework of Spatial Recurrence Plots (SRP) [Marwan2007Spatial](@cite). We give some examples about its use in [Spatial data](@ref).
- `AbstractGPUVector`: used for time series analysis with the GPU backend. We provide some explanations about it in [GPU](@ref).

!!! todo "Spatial Recurrence Microstates Analysis"
    RMA with SRP is an open research field. We include this functionality in the package for exploratory purposes, but the method is not yet mature enough for production use. Nevertheless, feel free to experiment with it in your research. 😃

```@docs
StateSpaceSet
```

### Output data from RecurrenceMicrostatesAnalysis.jl
When computing the RMA distribution, **RecurrenceMicrostatesAnalysis.jl** returns a [`Probabilities`](@ref).
This type is provided by **ComplexityMeasures.jl**, allowing this package to interoperate naturally with its tools and workflows.

```@docs
Probabilities
Counts
```