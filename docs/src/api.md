# API

### [Input data for RecurrenceMicrostatesAnalysis.jl](@id input_data)
**RecurrenceMicrostatesAnalysis.jl** accepts three types of input, each associated with a different backend:
- [`StateSpaceSet`](@ref) or `Vector{<:Real}`: used for multivariate time series, datasets, or state-space representations.
- `AbstractArray{<:Real}`: used for spatial data, enabling RMA to be applied within the generalized framework of Spatial Recurrence Plots (SRP) [Marwan2007Spatial](@cite). We give some examples about its use in [Spatial data](@ref).
- `AbstractGPUVector`: used for time series analysis with the GPU backend. We provide some explanations about it in GPU.

!!! todo "Spatial Recurrence Microstates Analysis"
    RMA with SRP is an open research field. We include this functionality in the package for exploratory purposes, but the method is not yet mature enough for production use. Nevertheless, feel free to experiment with it in your research. 😃

```@docs
StateSpaceSet
```

### [Output data from RecurrenceMicrostatesAnalysis.jl](@id output_data)
When computing the RMA distribution, **RecurrenceMicrostatesAnalysis.jl** returns a [`Probabilities`](@ref).
This type is provided by **ComplexityMeasures.jl**, allowing this package to interoperate naturally with its tools and workflows.

```@docs
Probabilities
Counts
```

## Recurrence Microstates
```@docs
RecurrenceMicrostates
```

### Recurrence expressions
```@docs
RecurrenceExpression
recurrence
ThresholdRecurrence
CorridorRecurrence
```

### Microstate shapes
```@docs
MicrostateShape
RectMicrostate
DiagonalMicrostate
TriangleMicrostate
```

### Sampling modes
```@docs
SamplingMode
SRandom
Full
```

### Sampling space
!!! todo "Future implementation"
    We pretend to expand the [`RecurrenceMicrostates`](@ref) structure to also consider
    a setted space from the recurrence plot as source of information to construct
    the RMA distribution.
    
```@docs
SamplingSpace
```

## Recurrence Quantification Analysis
```@docs
RecurrenceRate
RecurrenceDeterminism
RecurrenceLaminarity
Disorder
WindowedDisorder

rma
```

## Optimization
```@docs
Parameter
optimize
Threshold
```

## Operations
```@docs
Operation
operate
PermuteRows
PermuteColumns
Transpose
```

## Utils

### GPU Metrics
**Distances.jl** is not compatible with GPU execution; therefore, distance evaluations must be reimplemented
for GPU usage. This is done using a [`GPUMetric`](@ref).

```@docs
GPUMetric
GPUEuclidean
```