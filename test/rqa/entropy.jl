using Test
using Distributions
using RecurrenceMicrostatesAnalysis

x = StateSpaceSet(rand(Uniform(0, 1), 1000))
dist = probabilities(RecurrenceMicrostates(0.27, 4), x)

@test 0 ≤ measure(RecurrenceEntropy(), dist) ≤ log(length(dist))
@test 0 ≤ measure(RecurrenceEntropy(), x; N = 4) ≤ log(length(dist))

@test measure(RecurrenceEntropy(), dist) isa Real
@test measure(RecurrenceEntropy(), x; N = 4) isa Real