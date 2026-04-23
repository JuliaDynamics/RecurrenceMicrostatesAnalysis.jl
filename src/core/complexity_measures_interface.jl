
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