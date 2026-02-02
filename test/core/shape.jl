using Test
using RecurrenceMicrostatesAnalysis

@test_throws ArgumentError RecurrenceMicrostatesAnalysis.compute_motif(TestShape(), Standard(0.27), rand(2, 100), rand(2, 100), [1], [1], SVector{1, Int}(1))
@test_throws ArgumentError RecurrenceMicrostatesAnalysis.get_histogram_size(TestShape())
@test_throws ArgumentError RecurrenceMicrostatesAnalysis.get_power_vector(RecurrenceMicrostatesAnalysis.CPUCore(), TestShape())
@test_throws ArgumentError RecurrenceMicrostatesAnalysis.get_offsets(RecurrenceMicrostatesAnalysis.CPUCore(), TestShape())