using Test
using RecurrenceMicrostatesAnalysis

@test operate(PermuteColumns(RectMicrostate(3, 3)), 237, 2) == 347
@test operate(PermuteRows(RectMicrostate(3, 3)), 237, [1, 3, 2]) == 349
@test operate(Transpose(RectMicrostate(3, 3)), 237) == 231