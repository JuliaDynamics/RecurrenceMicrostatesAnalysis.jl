export RecurrenceMicrostates

##########################################################################################
#   Recurrence Microstate
##########################################################################################
"""
    RecurrenceMicrostates
"""
struct RecurrenceMicrostates{MS <: MicrostateShape, RE <: RecurrenceExpression, SM <: SamplingMode, C <: RMACore} <: CountBasedOutcomeSpace 
    shape::MS
    expr::RE
    sampling::SM
    core::C
end

##########################################################################################
#   Recurrence Microstate: Convenience constructors
##########################################################################################
function RecurrenceMicrostates(ε::Real, N::Int; sampling_ratio::Real = 0.05, core::RMACore = CPUCore())

    ##      If using GPU, assert Float32.
    if (core isa GPUCore)
        @assert ε isa Float32 "When using GPU, the threshold must be a Float32."
    end

    shape = Rect(N)
    sampling = SRandom(sampling_ratio)
    expr = Standard(ε)
    return RecurrenceMicrostates(shape, expr, sampling, core)
end

function RecurrenceMicrostates(ε::Real, shape::MicrostateShape; sampling_ratio::Real = 0.05, core::RMACore = CPUCore())

    ##      If using GPU, assert Float32.
    if (core isa GPUCore)
        @assert ε isa Float32 "When using GPU, the threshold must be a Float32."
    end

    sampling = SRandom(sampling_ratio)
    expr = Standard(ε)
    return RecurrenceMicrostates(shape, expr, sampling, core)
end