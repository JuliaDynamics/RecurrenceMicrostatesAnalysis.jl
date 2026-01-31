export GPUCore, StandardGPUCore

##########################################################################################
#   RMACore: GPU
##########################################################################################
"""
    GPUCore{B} <: RMACore

Type which represents the pipeline executed by **RecurrenceMicrostatesAnalysis.jl** on
graphical processing units.

It is initialized using:
```julia
GPUCore(backend)
```
Here, `backend` is the GPU device backend, e.g., `MetalBackend`, `CUDABackend`.
"""
struct GPUCore{B} <: RMACore 
    backend::B
end

##########################################################################################
#   Implementation: compute_motif
##########################################################################################
@inline function gpu_compute_motif(expr, x, y, i, j, power_vector, offset, n)
    index = zero(Int32)

    @inbounds begin
        for m in eachindex(power_vector)
            dw, dh = offset[m]
            @fastmath index += power_vector[m] * gpu_recurrence(expr, x, y, i + dw, j + dh, n)
        end
    end

    return @fastmath index + 1
end

##########################################################################################
#   Implementation: histogram
##########################################################################################
#   Based on time series: (GPU)
#.........................................................................................
function histogram(
    rmspace::RecurrenceMicrostates{MS, RE, SM, C},
    x::AbstractGPUVector{SVector{N, Float32}},
    y::AbstractGPUVector{SVector{N, Float32}} = x;
    groupsize::Int = 256
) where {N, MS <: MicrostateShape, RE <: RecurrenceExpression, SM <: SamplingMode, C <: GPUCore}

    #   Info
    space = SamplingSpace(rmspace.shape, x, y)
    samples = get_num_samples(rmspace.sampling, space)

    #   Allocate memory
    pv = get_power_vector(rmspace.core, rmspace.shape)
    offsets = get_offsets(rmspace.core, rmspace.shape)

    hist = KernelAbstractions.zeros(rmspace.core.backend, Int32, get_histogram_size(rmspace.shape))

    #   Call the kernel
    if rmspace.sampling isa Full
        gpu_rng = KernelAbstractions.zeros(rmspace.core.backend, Int32, 1)
        gpu_histogram!(rmspace.core.backend, groupsize)(x, y, pv, offsets, rmspace.core, rmspace.expr, rmspace.sampling, space, Int32(samples), hist, gpu_rng, Int32(N); ndrange = samples)
    else
        rng = get_sample(rmspace.core, rmspace.sampling, space, samples)
        gpu_rng = KernelAbstractions.zeros(rmspace.core.backend, SVector{2,Int32}, samples)
        copyto!(gpu_rng, rng)

        gpu_histogram!(rmspace.core.backend, groupsize)(x, y, pv, offsets, rmspace.core, rmspace.expr, rmspace.sampling, space, Int32(samples), hist, gpu_rng, Int32(N); ndrange = samples)
    end

    KernelAbstractions.synchronize(rmspace.core.backend)
    res =  hist |> Vector
    return res
end

##########################################################################################
#   Implementation: GPU Kernels
##########################################################################################
@kernel function gpu_histogram!(x, y, pv, offsets, core, expr, sampling, space, samples, hist, rng, n)
    m = @index(Global)
    if m <= samples
        i = zero(Int32)
        j = zero(Int32)

        if sampling isa Full
            i, j = get_sample(core, sampling, space, rng, m)
        else
            i = rng[m][1]
            j = rng[m][2]
        end

        idx = gpu_compute_motif(expr, x, y, i, j, pv, offsets, n)
        
        Atomix.@atomic hist[idx] += one(Int32)
    end
end
##########################################################################################
#   Implementation: distribution
##########################################################################################
distribution(
    x::AbstractGPUVector{SVector{N, Float32}},
    y::AbstractGPUVector{SVector{N, Float32}},
    shape::MicrostateShape;
    rate::Float32 = 0.05f0,
    sampling::SamplingMode = SRandom(rate),
    groupsize::Int = 256,
    backend = get_backend(x)
) where {N} = distribution(GPUCore(backend, shape, sampling), x, y; groupsize = groupsize)
#.........................................................................................
distribution(
    x::AbstractGPUVector{SVector{N, Float32}}, 
    y::AbstractGPUVector{SVector{N, Float32}},
    expr::RecurrenceExpression,
    n::Int;
    rate::Float32 = 0.05f0,
    sampling::SamplingMode = SRandom(rate),
    groupsize::Int = 256,
    backend = get_backend(x)
) where {N} = distribution(GPUCore(backend, Rect(expr, n), sampling), x, y; groupsize = groupsize)
#.........................................................................................
distribution(
    x::AbstractGPUVector{SVector{N, Float32}},
    y::AbstractGPUVector{SVector{N, Float32}},
    ε::Float32,
    n::Int;
    rate::Float32 = 0.05f0,
    sampling::SamplingMode = SRandom(rate),
    groupsize::Int = 256,
    backend = get_backend(x),
    metric::GPUMetric = GPUEuclidean()
) where {N} = distribution(x, y, Standard(ε; metric = metric), n; rate = rate, sampling = sampling, groupsize = groupsize, backend = backend)
#.........................................................................................
distribution(
    x::AbstractGPUVector{SVector{N, Float32}},
    shape::MicrostateShape;
    rate::Float32 = 0.05f0,
    sampling::SamplingMode = SRandom(rate),
    groupsize::Int = 256,
    backend = get_backend(x),
) where{N} = distribution(x, x, shape; rate = rate, sampling = sampling, groupsize = groupsize, backend = backend)
#.........................................................................................
distribution(
    x::AbstractGPUVector{SVector{N, Float32}},
    expr::RecurrenceExpression,
    n::Int;
    rate::Float32 = 0.05f0,
    sampling::SamplingMode = SRandom(rate),
    groupsize::Int = 256,
    backend = get_backend(x),
) where{N} = distribution(x, x, expr, n; rate = rate, sampling = sampling, groupsize = groupsize, backend = backend)
#.........................................................................................
distribution(
    x::AbstractGPUVector{SVector{N, Float32}},
    ε::Float32,
    n::Int;
    rate::Float32 = 0.05f0,
    sampling::SamplingMode = SRandom(rate),
    groupsize::Int = 256,
    backend = get_backend(x),
    metric::GPUMetric = GPUEuclidean()
) where {N} = distribution(x, x, ε, n; rate = rate, sampling = sampling, groupsize = groupsize, backend = backend, metric = metric)
#.........................................................................................
distribution(
    core::GPUCore,
    x;
    groupsize::Int = 256,
) = distribution(core, x, x; groupsize = groupsize)
#.........................................................................................
"""
    distribution(core::GPUCore, [x], [y]; kwargs...)

Compute an RMA distribution for the input data `[x]` and `[y]` using a GPU backend configuration
defined by `core`, which must be a [`GPUCore`](@ref).

The inputs `[x]` and `[y]` must be vectors of type `AbstractGPUVector`. This method supports
time-series analysis only.

!!! note
    The resulting distribution is copied from GPU memory back to the CPU.

### Arguments
- `core`: A [`GPUCore`](@ref) defining the [`MicrostateShape`](@ref),
  [`RecurrenceExpression`](@ref), and [`SamplingMode`](@ref) used in the computation.
- `[x]`: Input data provided as an `AbstractGPUVector`.
- `[y]`: Input data provided as an `AbstractGPUVector`.

### Keyword Arguments
- `groupsize`: Number of threads per GPU workgroup.

### Examples
```julia
using CUDA
gpudata = StateSpaceSet(Float32.(data)) |> CuVector
core = GPUCore(CUDABackend(), Rect(Standard(0.27f0; metric = GPUEuclidean()), 2), SRandom(0.05))
dist = distribution(core, gpudata, gpudata)
```

!!! warning
    Spatial data are not supported by [`GPUCore`](@ref).
"""
function distribution(
    core::GPUCore,
    x,
    y;
    groupsize::Int = 256,
)
    hist = histogram(core, x, y; groupsize = groupsize)
    return Probabilities(hist)
end

##########################################################################################