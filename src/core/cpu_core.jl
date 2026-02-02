##########################################################################################
#   RMACore: CPU
##########################################################################################
"""
    CPUCore <: RMACore

Type which represents the pipeline executed by **RecurrenceMicrostatesAnalysis.jl** on
central processing units.
"""
struct CPUCore <: RMACore end

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

function histogram(
    rmspace::RecurrenceMicrostates{MS, RE, SM},
    x::Union{StateSpaceSet, Vector{<: Real}},
    y::Union{StateSpaceSet, Vector{<: Real}} = x;
    threads = Threads.nthreads()
) where {MS <: MicrostateShape, RE <: RecurrenceExpression, SM <: SamplingMode}

    if (x isa Vector); x = StateSpaceSet(x); end
    if (y isa Vector); y = StateSpaceSet(y); end

    core = CPUCore()

    #   Info
    space = SamplingSpace(rmspace.shape, x, y)
    samples = get_num_samples(rmspace.sampling, space)

    #   Allocate memory
    pv = get_power_vector(core, rmspace.shape)
    offsets = get_offsets(core, rmspace.shape)

    #   Compute the histogram
    chunk = ceil(Int, samples / threads)
    tasks = Vector{Task}(undef, threads)

    for t in 1:threads
        tasks[t] = Threads.@spawn begin
            local_hist = zeros(Int, get_histogram_size(rmspace.shape))
            local_rng = TaskLocalRNG()

            start = (t - 1) * chunk + 1
            stop  = min(t * chunk, samples)

            for m in start:stop
                i, j = get_sample(core, rmspace.sampling, space, local_rng, m)
                idx = compute_motif(rmspace.expr, x, y, i, j, pv, offsets)
                @inbounds local_hist[idx] += 1
            end

            return local_hist
        end
    end

    res = reduce(+, fetch.(tasks))
    return res
end
#.........................................................................................
#   Based on spatial data: (CPU only)
#.........................................................................................
function histogram(
    rmspace::RecurrenceMicrostates{MS, RE, SM},
    x::AbstractArray{<: Real},
    y::AbstractArray{<: Real} = x;
    threads = Threads.nthreads()
) where {MS <: MicrostateShape, RE <: RecurrenceExpression, SM <: SamplingMode}
    #   Info
    space = SamplingSpace(rmspace.shape, x, y)
    samples = get_num_samples(rmspace.sampling, space)
    dim_x = ndims(x) - 1
    dim_y = ndims(y) - 1

    core = CPUCore()

    #   Allocate memory
    pv = get_power_vector(core, rmspace.shape)

    #   Compute the histogram
    chunk = ceil(Int, samples / threads)
    tasks = Vector{Task}(undef, threads)

    for t in 1:threads
        tasks[t] = Threads.@spawn begin
            local_hist = zeros(Int, get_histogram_size(rmspace.shape))
            local_rng = TaskLocalRNG()

            start = (t - 1) * chunk + 1
            stop  = min(t * chunk, samples)

            idx = zeros(Int, dim_x + dim_y)
            itr = zeros(Int, dim_x + dim_y)

            for m in start:stop
                get_sample(core, rmspace.sampling, space, idx, local_rng, m)
                i = compute_motif(rmspace.shape, rmspace.expr, x, y, idx, itr, pv)
                @inbounds local_hist[i] += 1
            end

            return local_hist
        end
    end

    res =  reduce(+, fetch.(tasks))
    return res
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