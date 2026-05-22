##########################################################################################
#   RMACore: GPU
##########################################################################################
struct GPUCore <: RMACore end

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
    rmspace::RecurrenceMicrostates{MS, <: RecurrenceExpression{T, M}, SM, SS},
    x::AbstractGPUVector{SVector{N, T}},
    y::AbstractGPUVector{SVector{N, T}} = x;
    groupsize::Int = 256
) where {MS <: MicrostateShape, SM <: SamplingMode, SS <: AbstractSamplingSpace, N, T <: Real, M <: GPUMetric}

    #   Get backend.
    backend = KernelAbstractions.get_backend(x)
    core = GPUCore()

    #   Info
    space = SamplingSpace(rmspace.shape, rmspace.space, x, y)
    samples = get_num_samples(rmspace.sampling, space)

    #   Allocate memory
    pv = get_power_vector(core, rmspace.shape)
    offsets = get_offsets(core, rmspace.shape)

    hist = KernelAbstractions.zeros(backend, Int32, get_histogram_size(rmspace.shape))

    #   Call the kernel
    if rmspace.sampling isa Full
        gpu_rng = KernelAbstractions.zeros(backend, Int32, 1)
        gpu_histogram!(backend, groupsize)(x, y, pv, offsets, core, rmspace.expr, rmspace.sampling, space, Int32(samples), hist, gpu_rng, Int32(N); ndrange = samples)
    else
        rng = get_sample(core, rmspace.sampling, space, samples)
        gpu_rng = KernelAbstractions.zeros(backend, SVector{2,Int32}, samples)
        copyto!(gpu_rng, rng)

        gpu_histogram!(backend, groupsize)(x, y, pv, offsets, core, rmspace.expr, rmspace.sampling, space, Int32(samples), hist, gpu_rng, Int32(N); ndrange = samples)
    end

    KernelAbstractions.synchronize(backend)
    res::Vector{Int} =  hist |> Vector
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