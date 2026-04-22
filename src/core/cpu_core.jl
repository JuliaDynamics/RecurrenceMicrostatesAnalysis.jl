##########################################################################################
#   RMACore: CPU
##########################################################################################
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
    threads::Int = 0
) where {MS <: MicrostateShape, RE <: RecurrenceExpression, SM <: SamplingMode}

    #   Threads, input and core
    threads = threads <= 0 ? Threads.nthreads() : threads
    x_input = x isa Vector ? x |> StateSpaceSet : x
    y_input = y isa Vector ? y |> StateSpaceSet : y
    core = CPUCore()

    #   Info
    space = SamplingSpace(rmspace.shape, x_input, y_input)
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
                idx = compute_motif(rmspace.expr, x_input, y_input, i, j, pv, offsets)
                @inbounds local_hist[idx] += 1
            end

            return local_hist
        end
    end

    res::Vector{Int} = reduce(+, fetch.(tasks))
    return res
end
#.........................................................................................
#   Based on spatial data: (CPU only)
#.........................................................................................
function histogram(
    rmspace::RecurrenceMicrostates{MS, RE, SM},
    x::AbstractArray{<: Real},
    y::AbstractArray{<: Real} = x;
    threads::Int = 0
) where {MS <: MicrostateShape, RE <: RecurrenceExpression, SM <: SamplingMode}
    #   Core and threads
    threads = threads <= 0 ? Threads.nthreads() : threads
    core = CPUCore()

    #   Info
    space = SamplingSpace(rmspace.shape, x, y)
    samples = get_num_samples(rmspace.sampling, space)
    dim_x = ndims(x) - 1
    dim_y = ndims(y) - 1

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

    res::Vector{Int} =  reduce(+, fetch.(tasks))
    return res
end

##########################################################################################