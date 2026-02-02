using Test
using Distributions
using RecurrenceMicrostatesAnalysis

x = StateSpaceSet(rand(1000))
dist = probabilities(RecurrenceMicrostates(0.27, 4), x)

@test 0 ≤ measure(RecurrenceRate(), dist) ≤ 1
@test 0 ≤ measure(RecurrenceRate(), x) ≤ 1

@test measure(RecurrenceRate(), dist) isa Real
@test measure(RecurrenceRate(), x) isa Real