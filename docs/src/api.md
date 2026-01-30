# API

# ## Probabilities and counts

```@docs
RecurrenceMicrostates
RecurrenceExpression
CorridorRecurrence
ThresholdRecurrence # What uised to be called Standard
```

# ## Specification of recurrence microstates

```@docs
RecurrenceExpression
MicrostateShape
```




Alternatively, a [`RecurrenceExpression`](@ref) can be specified directly:
```julia
distribution([x], expr::RecurrenceExpression, n::Int; kwargs...)
```
**Example:**
```@example quick_example
expr = Corridor(0.05, 0.27)
dist = distribution(ssset, expr, 2)
```

If a custom [`MicrostateShape`](@ref) is required, the call simplifies to:
```julia
distribution([x], shape::MicrostateShape; kwargs...)
```
**Example:**
```@example quick_example
shape = Triangle(Standard(0.27), 3)
dist = distribution(ssset, shape)
```


# ## Sampling modes

```@docs
SRandom
what else?
```

# ## Computational specification

```@docs
CPUCore
GPUCore
```

The following needs to change:

```
This method automatically selects a [`CPUCore`](@ref) when `x` is a [`StateSpaceSet`](@ref)
and a [`GPUCore`](@ref) when `x` is an `AbstractGPUVector`. By default, square microstates of size `n` are used.

Additional keyword arguments include:
- `rate::Float64`: Sampling rate (default: `0.05`).
- `sampling::SamplingMode`: Sampling mode (default: [`SRandom`](@ref)).
- `metric::Metric`: Distance metric from [Distances.jl](https://github.com/JuliaStats/Distances.jl). When using a [`GPUCore`](@ref), a [`GPUMetric`](@ref) must be provided.
```

all these keyword arguments should not be given to the
`probabilities` function. THey should be given to a generic computation type that is a field of the RecurrenceMicrostates!!!

The following must be in the GPUCore struct:

!!! warning
    GPU backends require inputs of type `Float32`. `Float64` inputs are not supported on GPU.



# ##