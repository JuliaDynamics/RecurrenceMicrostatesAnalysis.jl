using Test
using RecurrenceMicrostatesAnalysis

struct TestShape <: MicrostateShape end
struct TestSampling <: SamplingMode end

@test_throws ArgumentError SamplingSpace(TestShape(), rand(100) |> StateSpaceSet, rand(100) |> StateSpaceSet)
@test_throws ArgumentError SamplingSpace(TestShape(), rand(2, 100), rand(2, 100))

space = SamplingSpace(RectMicrostate(2), rand(100) |> StateSpaceSet, rand(100) |> StateSpaceSet)
@test_throws ArgumentError RecurrenceMicrostatesAnalysis.get_sample(RecurrenceMicrostatesAnalysis.CPUCore(), TestSampling(), space)
@test_throws ArgumentError RecurrenceMicrostatesAnalysis.get_num_samples(TestSampling(), space)