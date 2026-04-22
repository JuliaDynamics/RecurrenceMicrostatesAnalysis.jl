
##########################################################################################
#   Interface with ComplexityMeasures.jl
##########################################################################################
function ComplexityMeasures.counts_and_outcomes(rmspace::RecurrenceMicrostates, x, y = x)
    counts = histogram(rmspace, x, y)
    outcomes = eachindex(counts)
    return counts, outcomes
end

function ComplexityMeasures.codify(rmspace::RecurrenceMicrostates, x, y = x)
    return histogram(rmspace, x, y)
end

function ComplexityMeasures.outcome_space(rmspace::RecurrenceMicrostates, x, y = x)
    return eachindex(1:get_histogram_size(rmspace.shape))
end

#
#      Needed to CRP
function ComplexityMeasures.probabilities(o::RecurrenceMicrostates, x, y)
    return first(probabilities_and_outcomes(o, x, y))
end

function ComplexityMeasures.probabilities_and_outcomes(o::RecurrenceMicrostates, x, y)
    cts, outs = counts_and_outcomes(o, x, y)
    probs = Probabilities(cts, outs)
    return probs, outcomes(probs)
end

#=
# Define outcome space

struct RecurrenceMicrostates{MS<:MicrostateShape, RE <: RecurrenceExpression, SM<:SamplingMode, C<:Core} <: CountBasedOutcomeSpace
    shape::MS
    expression::RE
    sampling::SM
    core::C # cpu or gpu
    what_else # ?
end

# convenience constructors

function RecurrenceMicrostates(;
        ε = nothing,
        N = 2,
        core = CPUCore(),
        whatever_else
    )

    # create the data structure
    RecurrenceMicrostates(...)
end


# downstream convenience
function compute_motif(ospace::RecurrenceMicrostates, x, y = x)
    compute_motif(ospace.expression, ..., x, y)
end


# Define necessary functions to actually make it an `OutcomeSpace`

function ComplexityMeasures.counts_and_outcomes(rmspace::RecurrenceMicrostates, x, y = x)
    a, b, c = extract_defining_types(rmspace)
    counts = histogram(a,b,c, x, y) # Fix this of course
    outcomes = xxx # somehow must generate them
    return Counts()
end

function ComplexityMeasures.codify(rmspace::RecurrenceMicrostates, x, y = x)
    # TODO
end

function ComplexityMeasures.outcome_space(rmspace::RecurrenceMicrostates, x, y = x)
    # TODO
end
=#

# The rest is taken care of by ComplexityMeasures.jl. Including `entropy(...)`.

# TODOs:

# DONE - all microstate shape types need to be renamed; they conflict with
# plotting packages too much.
#
# Furthermore, they should be orthogonal inputs to the main outcome space t