using Test
using RecurrenceMicrostatesAnalysis

struct TestExpression <: RecurrenceExpression{Float64, Euclidean} end

@test_throws ArgumentError recurrence(TestExpression(), rand(100) |> StateSpaceSet, rand(100) |> StateSpaceSet, 1, 1)
@test_throws ArgumentError recurrence(TestExpression(), rand(2, 100), rand(2, 100), (1, ), (1, ))