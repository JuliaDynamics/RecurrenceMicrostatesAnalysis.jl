export RecurrenceMicrostates

##########################################################################################
#   Recurrence Microstate
##########################################################################################
"""
    RecurrenceMicrostates
"""
struct RecurrenceMicrostates{MS <: MicrostateShape, RE <: RecurrenceExpression, SM <: SamplingMode, C <: RMACore} <: ComplexityMeasures.CountBasedOutcomeSpace 
    shape::MS
    expr::RE
    sampling::SM
    core::C
end

##########################################################################################
#   Recurrence Microstate: Convenience constructors
##########################################################################################
function RecurrenceMicrostates(ε::Real, N::Int; sampling_ratio::Real = 0.05, sampling::SamplingMode = SRandom(sampling_ratio), core::RMACore = CPUCore(), metric::Metric = DEFAULT_METRIC)

    ##      If using GPU, assert Float32.
    if (core isa GPUCore)
        @assert ε isa Float32 "When using GPU, the threshold must be a Float32."
        if (!(metric isa GPUMetric))
            println("Warning: GPU backend must use a GPUMetric to evaluate distance. It will use a GPUEuclidean to avoid errors.")
            metric = GPUEuclidean()
        end
    end

    shape = RectMicrostate(N)
    expr = Standard(ε; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling, core)
end

function RecurrenceMicrostates(ε::Real, structure::NTuple; sampling_ratio::Real = 0.05, sampling::SamplingMode = SRandom(sampling_ratio), metric::Metric = DEFAULT_METRIC)
    shape = RectMicrostate(structure)
    expr = Standard(ε; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling, CPUCore())
end

function RecurrenceMicrostates(ε::Real, shape::MicrostateShape; sampling_ratio::Real = 0.05, sampling::SamplingMode = SRandom(sampling_ratio), core::RMACore = CPUCore(), metric::Metric = DEFAULT_METRIC)

    ##      If using GPU, assert Float32.
    if (core isa GPUCore)
        @assert ε isa Float32 "When using GPU, the threshold must be a Float32."
        if (!(metric isa GPUMetric))
            println("Warning: GPU backend must use a GPUMetric to evaluate distance. It will use a GPUEuclidean to avoid errors.")
            metric = GPUEuclidean()
        end
    end

    expr = Standard(ε; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling, core)
end

function RecurrenceMicrostates(ε_min::Real, ε_max::Real, N::Int; sampling_ratio::Real = 0.05, sampling::SamplingMode = SRandom(sampling_ratio), core::RMACore = CPUCore(), metric::Metric = DEFAULT_METRIC)

    ##      If using GPU, assert Float32.
    if (core isa GPUCore)
        @assert ε isa Float32 "When using GPU, the threshold must be a Float32."
        if (!(metric isa GPUMetric))
            println("Warning: GPU backend must use a GPUMetric to evaluate distance. It will use a GPUEuclidean to avoid errors.")
            metric = GPUEuclidean()
        end
    end

    shape = RectMicrostate(N)
    expr = Corridor(ε_min, ε_max; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling, core)
end

function RecurrenceMicrostates(ε_min::Real, ε_max::Real, structure::NTuple; sampling_ratio::Real = 0.05, sampling::SamplingMode = SRandom(sampling_ratio), metric::Metric = DEFAULT_METRIC)
    shape = RectMicrostate(structure)
    expr = Corridor(ε_min, ε_max; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling, CPUCore())
end

function RecurrenceMicrostates(ε_min::Real, ε_max::Real, shape::MicrostateShape; sampling_ratio::Real = 0.05, sampling::SamplingMode = SRandom(sampling_ratio), core::RMACore = CPUCore(), metric::Metric = DEFAULT_METRIC)

    ##      If using GPU, assert Float32.
    if (core isa GPUCore)
        @assert ε isa Float32 "When using GPU, the threshold must be a Float32."
        if (!(metric isa GPUMetric))
            println("Warning: GPU backend must use a GPUMetric to evaluate distance. It will use a GPUEuclidean to avoid errors.")
            metric = GPUEuclidean()
        end
    end

    expr = Corridor(ε_min, ε_max; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling, core)
end

##########################################################################################