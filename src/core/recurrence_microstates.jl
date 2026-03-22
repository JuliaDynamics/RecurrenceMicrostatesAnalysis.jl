export RecurrenceMicrostates

##########################################################################################
#   Recurrence Microstate
##########################################################################################
# TODO: Write RecurrenceMicrostates documentation.
"""
    RecurrenceMicrostates
"""
struct RecurrenceMicrostates{MS <: MicrostateShape, RE <: RecurrenceExpression, SM <: SamplingMode} <: ComplexityMeasures.CountBasedOutcomeSpace
    shape::MS
    expr::RE
    sampling::SM
end

##########################################################################################
#   Recurrence Microstate: Convenience constructors
##########################################################################################
function RecurrenceMicrostates(expr::RecurrenceExpression, N::Int; sampling_ratio::Real = 0.05, sampling::SamplingMode = SRandom(sampling_ratio))
    shape = RectMicrostate(N)
    return RecurrenceMicrostates(shape, expr, sampling)
end

function RecurrenceMicrostates(ε::Real, N::Int; sampling_ratio::Real = 0.05, sampling::SamplingMode = SRandom(sampling_ratio), metric::Metric = DEFAULT_METRIC)
    shape = RectMicrostate(N)
    expr = Standard(ε; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling)
end

function RecurrenceMicrostates(expr::RecurrenceExpression, structure::NTuple; sampling_ratio::Real = 0.05, sampling::SamplingMode = SRandom(sampling_ratio))
    shape = RectMicrostate(structure)
    return RecurrenceMicrostates(shape, expr, sampling)
end

function RecurrenceMicrostates(ε::Real, structure::NTuple; sampling_ratio::Real = 0.05, sampling::SamplingMode = SRandom(sampling_ratio), metric::Metric = DEFAULT_METRIC)
    shape = RectMicrostate(structure)
    expr = Standard(ε; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling)
end

function RecurrenceMicrostates(expr::RecurrenceExpression, shape::MicrostateShape; sampling_ratio::Real = 0.05, sampling::SamplingMode = SRandom(sampling_ratio))
    return RecurrenceMicrostates(shape, expr, sampling)
end

function RecurrenceMicrostates(ε::Real, shape::MicrostateShape; sampling_ratio::Real = 0.05, sampling::SamplingMode = SRandom(sampling_ratio), metric::Metric = DEFAULT_METRIC)
    expr = Standard(ε; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling)
end

function RecurrenceMicrostates(ε_min::Real, ε_max::Real, N::Int; sampling_ratio::Real = 0.05, sampling::SamplingMode = SRandom(sampling_ratio), metric::Metric = DEFAULT_METRIC)
    shape = RectMicrostate(N)
    expr = Corridor(ε_min, ε_max; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling)
end

function RecurrenceMicrostates(ε_min::Real, ε_max::Real, structure::NTuple; sampling_ratio::Real = 0.05, sampling::SamplingMode = SRandom(sampling_ratio), metric::Metric = DEFAULT_METRIC)
    shape = RectMicrostate(structure)
    expr = Corridor(ε_min, ε_max; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling)
end

function RecurrenceMicrostates(ε_min::Real, ε_max::Real, shape::MicrostateShape; sampling_ratio::Real = 0.05, sampling::SamplingMode = SRandom(sampling_ratio), metric::Metric = DEFAULT_METRIC)
    expr = Corridor(ε_min, ε_max; metric = metric)
    return RecurrenceMicrostates(shape, expr, sampling)
end
