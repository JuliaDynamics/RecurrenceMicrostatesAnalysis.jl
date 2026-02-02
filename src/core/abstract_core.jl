export histogram

##########################################################################################
#   RMACore
##########################################################################################
"""
    RMACore

Abstract supertype that defines the execution pipeline of the package.

An instance of **RMACore** must be provided to the [`histogram`](@ref) function to determine how the
histogram computation is performed.

Concrete implementations of `RMACore` are [`CPUCore`](@ref) and [`GPUCore`](@ref), which target CPU
and GPU execution, respectively. Implementing custom subtypes of `RMACore` is **strongly discouraged**,
as doing so requires reimplementing several internal utilities for the package ecosystem to function
correctly.

#   Implementations
- [`CPUCore`](@ref)
- [`GPUCore`](@ref)
"""
abstract type RMACore end

##########################################################################################
#   Implementations: histogram
##########################################################################################
"""
    histogram(rmspace::RecurrenceMicrostates, [x], [y]; kwargs...)

Compute the histogram of recurrence microstates for an abstract recurrence structure constructed
from the input data `[x]` and `[y]`.

If `[x]` and `[y]` are identical, the result corresponds to a Recurrence Plot (RP); otherwise, it
corresponds to a Cross-Recurrence Plot (CRP).

The result is returned as a `Vector{Int}` representing the histogram of recurrence
microstates for the given input data.

### Arguments
- `rmspace`: A [`RecurrenceMicrostates`](@ref) which defines the outcome space.
- `[x]`: Input data provided as a [`StateSpaceSet`](@ref) or an `AbstractArray`. If using a [`GPUCore`](@ref) the input must be an `AbstractGPUVector`.
- `[y]`: Input data provided as a [`StateSpaceSet`](@ref) or an `AbstractArray`. If using a [`GPUCore`](@ref) the input must be an `AbstractGPUVector`.

!!! note
    [`StateSpaceSet`](@ref) and `AbstractArray` inputs use different internal backends and therefore
    different histogram implementations. Both interfaces share the same method signature, differing
    only in the input data representation.

!!! info
    We strongly recommend to use [`StateSpaceSet`](@ref) as input for time series. However, if given
    a `Vector{Real}` as input it is accepted and converted internally to a [`StateSpaceSet`](@ref).

### Keyword Arguments
If using a [`CPUCore`](@ref):
- `threads`: Number of threads used to compute the histogram. By default, this is set to
  `Threads.nthreads()`, which can be specified at Julia startup using `--threads N` or via the
  `JULIA_NUM_THREADS` environment variable.

If using a [`GPUCore`](@ref):
- `groupsize`: Number of threads per GPU workgroup.

### Examples using [`CPUCore`](@ref)
- Time series:
```julia
ssset = StateSpaceSet(rand(Float64, (1000)))
rmspace = RecurrenceMicrostates(0.27, 3)
dist = histogram(rmspace, ssset)
```

- Spatial data:
```julia
spatialdata = rand(Float64, (3, 50, 50))
rmspace = RecurrenceMicrostates(0.27, RectMicrostate((2, 1, 2, 1)))
dist = histogram(rmspace, spatialdata)
```

### Examples using [`GPUCore`](@ref)
```julia
using CUDA
gpudata = StateSpaceSet(Float32.(data)) |> CuVector
core = GPUCore(CUDABackend(), Rect(Standard(0.27f0; metric = GPUEuclidean()), 2), SRandom(0.05))
dist = histogram(core, gpudata, gpudata)
```

!!! note
    The resulting histogram is copied from GPU memory back to the CPU.
"""
function histogram() 
    throw(ArgumentError("`histogram` is not implemented without arguments."))
end

##########################################################################################