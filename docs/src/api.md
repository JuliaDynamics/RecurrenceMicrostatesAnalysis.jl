# API

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