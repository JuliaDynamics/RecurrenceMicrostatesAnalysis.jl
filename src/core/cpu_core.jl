export CPUCore, StandardCPUCore

##########################################################################################
#   RMACore: CPU
##########################################################################################
"""
    CPUCore{M<:MicrostateShape, S<:SamplingMode} <: RMACore

Abstract CPU backend that implements the **RecurrenceMicrostatesAnalysis.jl** execution pipeline on
central processing units.

The package provides a default implementation via [`StandardCPUCore`](@ref).

Concrete subtypes of `CPUCore` must define the following fields:
- `shape`: the [`MicrostateShape`](@ref) used to construct microstates.
- `sampling`: the [`SamplingMode`](@ref) used to sample the recurrence space.

#   Implementations
- [`StandardCPUCore`](@ref)
"""
abstract type CPUCore{M<:MicrostateShape, S<:SamplingMode} <: RMACore end
#.........................................................................................
"""
    StandardCPUCore{M<:MicrostateShape, S<:SamplingMode} <: CPUCore{M, S}

Default CPU backend implementation for **RecurrenceMicrostatesAnalysis.jl**.

This type provides the standard execution pipeline for computing recurrence microstate distributions
on CPU devices.

#   Initialization
```julia
core = CPUCore(shape, sampling)
```
"""
struct StandardCPUCore{M<:MicrostateShape, S<:SamplingMode} <: CPUCore{M, S}
    shape::M
    sampling::S
end


#.........................................................................................
CPUCore(shape::M, sampling::S) where {M<:MicrostateShape, S<:SamplingMode} = StandardCPUCore(shape, sampling)

##########################################################################################
#   Implementation: compute_motif
##########################################################################################
@inline function compute_motif(
    expr::E,
    x::StateSpaceSet,
    y::StateSpaceSet,
    i::Int,
    j::Int,
    power_vector::SVector{D, Int},
    offsets::SVector{D, SVector{2, Int}}
) where {E<:RecurrenceExpression, D}
    @inbounds begin
        index = 0

        for m in eachindex(power_vector)
            dw, dh = offsets[m]
            @fastmath index += recurrence(expr, x, y, i + dw, j + dh) * power_vector[m]
        end

        return index + 1
    end
end

##########################################################################################
#   Implementation: histogram
##########################################################################################
#   Based on time series: (CPU)
#.........................................................................................
"""
    histogram(core::StandardCPUCore, [x], [y]; kwargs...)

Compute the histogram of recurrence microstates for an abstract recurrence structure constructed
from the input data `[x]` and `[y]`.

If `[x]` and `[y]` are identical, the result corresponds to a Recurrence Plot (RP); otherwise, it
corresponds to a Cross-Recurrence Plot (CRP).

The result is returned as a [`Counts`](@ref) object representing the histogram of recurrence
microstates for the given input data.

This method implements the CPU backend using a [`CPUCore`](@ref), specifically the
[`StandardCPUCore`](@ref) implementation.

### Arguments
- `core`: A [`StandardCPUCore`](@ref) defining the CPU backend configuration.
- `[x]`: Input data provided as a [`StateSpaceSet`](@ref) or an `AbstractArray`.
- `[y]`: Input data provided as a [`StateSpaceSet`](@ref) or an `AbstractArray`.

!!! note
    [`StateSpaceSet`](@ref) and `AbstractArray` inputs use different internal backends and therefore
    different histogram implementations. Both interfaces share the same method signature, differing
    only in the input data representation.

### Keyword Arguments
- `threads`: Number of threads used to compute the histogram. By default, this is set to
  `Threads.nthreads()`, which can be specified at Julia startup using `--threads N` or via the
  `JULIA_NUM_THREADS` environment variable.

### Examples
- Time series:
```julia
ssset = StateSpaceSet(rand(Float64, (1000)))
core = CPUCore(Rect(Standard(0.27), 2), SRandom(0.05))
dist = histogram(core, ssset, ssset)
```

- Spatial data:
```julia
spatialdata = rand(Float64, (3, 50, 50))
core = CPUCore(Rect(Standard(0.5), (2, 2, 1, 1)), SRandom(0.05))
dist = histogram(core, spatialdata, spatialdata)
```
"""
function histogram(
    core::StandardCPUCore,
    x::StateSpaceSet,
    y::StateSpaceSet;
    threads = Threads.nthreads()
)
    #   Info
    space = SamplingSpace(core.shape, x, y)
    samples = get_num_samples(core.sampling, space)

    #   Allocate memory
    pv = get_power_vector(core, core.shape)
    offsets = get_offsets(core, core.shape)

    #   Compute the histogram
    chunk = ceil(Int, samples / threads)
    tasks = Vector{Task}(undef, threads)

    for t in 1:threads
        tasks[t] = Threads.@spawn begin
            local_hist = zeros(Int, get_histogram_size(core.shape))
            local_rng = TaskLocalRNG()

            start = (t - 1) * chunk + 1
            stop  = min(t * chunk, samples)

            for m in start:stop
                i, j = get_sample(core, core.sampling, space, local_rng, m)
                idx = compute_motif(core.shape.expr, x, y, i, j, pv, offsets)
                @inbounds local_hist[idx] += 1
            end

            return local_hist
        end
    end

    res = reduce(+, fetch.(tasks))
    out = eachindex(res)

    return Counts(res, out)
end
#.........................................................................................
#   Based on spatial data: (CPU only)
#.........................................................................................
function histogram(
    core::StandardCPUCore,
    x::AbstractArray{<: Real},
    y::AbstractArray{<: Real};
    threads = Threads.nthreads()
)
    #   Info
    space = SamplingSpace(core.shape, x, y)
    samples = get_num_samples(core.sampling, space)
    dim_x = ndims(x) - 1
    dim_y = ndims(y) - 1

    #   Allocate memory
    pv = get_power_vector(core, core.shape)

    #   Compute the histogram
    chunk = ceil(Int, samples / threads)
    tasks = Vector{Task}(undef, threads)

    for t in 1:threads
        tasks[t] = Threads.@spawn begin
            local_hist = zeros(Int, get_histogram_size(core.shape))
            local_rng = TaskLocalRNG()

            start = (t - 1) * chunk + 1
            stop  = min(t * chunk, samples)

            idx = zeros(Int, dim_x + dim_y)
            itr = zeros(Int, dim_x + dim_y)

            for m in start:stop
                get_sample(core, core.sampling, space, idx, local_rng, m)
                i = compute_motif(core.shape, x, y, idx, itr, pv)
                @inbounds local_hist[i] += 1
            end

            return local_hist
        end
    end

    res =  reduce(+, fetch.(tasks))
    out = eachindex(res)

    return Counts(res, out)
end

##########################################################################################
#   Implementation: distribution
##########################################################################################
distribution(
    x::StateSpaceSet,
    y::StateSpaceSet,
    shape::MicrostateShape;
    rate::Float64 = 0.05,
    sampling::SamplingMode = SRandom(rate),
    threads::Int = Threads.nthreads()
) = distribution(CPUCore(shape, sampling), x, y; threads = threads)
#.........................................................................................
distribution(
    x::StateSpaceSet,
    y::StateSpaceSet,
    expr::RecurrenceExpression,
    n::Int;
    rate::Float64 = 0.05,
    sampling::SamplingMode = SRandom(rate),
    threads::Int = Threads.nthreads()
) = distribution(CPUCore(Rect(expr, n), sampling), x, y; threads = threads)
#.........................................................................................
distribution(
    x::StateSpaceSet,
    y::StateSpaceSet,
    ε::Float64,
    n::Int;
    rate::Float64 = 0.05,
    sampling::SamplingMode = SRandom(rate),
    threads::Int = Threads.nthreads(),
    metric::Metric = DEFAULT_METRIC
) = distribution(x, y, Standard(ε; metric = metric), n; rate = rate, sampling = sampling, threads = threads)
#.........................................................................................
distribution(
    x::StateSpaceSet,
    shape::MicrostateShape;
    rate::Float64 = 0.05,
    sampling::SamplingMode = SRandom(rate),
    threads::Int = Threads.nthreads()
) = distribution(x, x, shape; rate = rate, sampling = sampling, threads = threads)
#.........................................................................................
distribution(
    x::StateSpaceSet,
    expr::RecurrenceExpression,
    n::Int;
    rate::Float64 = 0.05,
    sampling::SamplingMode = SRandom(rate),
    threads::Int = Threads.nthreads()
) = distribution(x, x, expr, n; rate = rate, sampling = sampling, threads = threads)
#.........................................................................................
distribution(
    x::StateSpaceSet,
    ε::Float64,
    n::Int;
    rate::Float64 = 0.05,
    sampling::SamplingMode = SRandom(rate),
    threads::Int = Threads.nthreads(),
    metric::Metric = DEFAULT_METRIC
) = distribution(x, x, ε, n; rate = rate, sampling = sampling, threads = threads, metric = metric)
#.........................................................................................
distribution(
    x::AbstractArray{<:Real},
    y::AbstractArray{<:Real},
    shape::MicrostateShape;
    rate::Float64 = 0.05,
    sampling::SamplingMode = SRandom(rate),
    threads::Int = Threads.nthreads()
) = distribution(CPUCore(shape, sampling), x, y; threads = threads)
#.........................................................................................
distribution(
    x::AbstractArray{<:Real},
    shape::MicrostateShape;
    rate::Float64 = 0.05,
    sampling::SamplingMode = SRandom(rate),
    threads::Int = Threads.nthreads()
) = distribution(x, x, shape; rate = rate, sampling = sampling, threads = threads)
#.........................................................................................
distribution(
    core::CPUCore,
    x;
    threads = Threads.nthreads()
) = distribution(core, x, x; threads = threads)
#.........................................................................................
"""
    distribution(core::CPUCore, [x], [y]; kwargs...)

Compute an RMA distribution for the input data `[x]` and `[y]` using a CPU backend configuration
defined by `core`, which must be a [`CPUCore`](@ref).

For time-series analysis, the inputs `[x]` and `[y]` must be provided as [`StateSpaceSet`](@ref)
objects. For spatial analysis, the inputs must be provided as `AbstractArray`s.

### Arguments
- `core`: A [`CPUCore`](@ref) defining the [`MicrostateShape`](@ref),
  [`RecurrenceExpression`](@ref), and [`SamplingMode`](@ref) used in the computation.
- `[x]`: Input data provided as a [`StateSpaceSet`](@ref) or an `AbstractArray`.
- `[y]`: Input data provided as a [`StateSpaceSet`](@ref) or an `AbstractArray`.

### Keyword Arguments
- `threads`: Number of threads used to compute the distribution. By default, this is set to
  `Threads.nthreads()`, which can be specified at Julia startup using `--threads N` or via the
  `JULIA_NUM_THREADS` environment variable.

### Examples
- Time series:
```julia
ssset = StateSpaceSet(rand(Float64, (1000)))
core = CPUCore(Rect(Standard(0.27), 2), SRandom(0.05))
dist = distribution(core, ssset, ssset)
```

- Spatial data:
```julia
spatialdata = rand(Float64, (3, 50, 50))
core = CPUCore(Rect(Standard(0.5), (2, 2, 1, 1)), SRandom(0.05))
dist = distribution(core, spatialdata, spatialdata)
```
"""
function distribution(
    core::CPUCore,
    x,
    y;
    threads = Threads.nthreads()
)
    hist = histogram(core, x, y; threads = threads)
    return Probabilities(hist)
end
##########################################################################################