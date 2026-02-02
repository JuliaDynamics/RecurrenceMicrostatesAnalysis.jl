using Test
using RecurrenceMicrostatesAnalysis

@test_throws ArgumentError operate(PermuteColumns(RectMicrostate(3)))
@test_throws ArgumentError operate(PermuteRows(RectMicrostate(3)))
@test_throws ArgumentError operate(Transpose(RectMicrostate(3)))