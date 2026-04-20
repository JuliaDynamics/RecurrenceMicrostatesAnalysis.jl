using Test
using Distances: Euclidean
using RecurrenceMicrostatesAnalysis

struct TestParameter <: Parameter end
struct TestGPUMetric <: GPUMetric end
struct TestShape <: MicrostateShape end
struct TestSampling <: SamplingMode end
struct TestExpression <: RecurrenceExpression{Float64, Euclidean} end

@testset "GPU Metric structure" begin
    data = SVector{1, Float32}(0.0f0)

    @test_throws ArgumentError RecurrenceMicrostatesAnalysis.gpu_evaluate(TestGPUMetric(), data, data)
end

@testset "Operations structure" begin
    @test_throws ArgumentError operate(PermuteColumns(RectMicrostate(3)))
    @test_throws ArgumentError operate(PermuteRows(RectMicrostate(3)))
    @test_throws ArgumentError operate(Transpose(RectMicrostate(3)))
end

@testset "Optimize structure" begin
    @test_throws ArgumentError optimize(TestParameter())
end

@testset "Recurrence structure" begin
    @test_throws ArgumentError recurrence(TestExpression(), rand(100) |> StateSpaceSet, rand(100) |> StateSpaceSet, 1, 1)
    @test_throws ArgumentError recurrence(TestExpression(), rand(2, 100), rand(2, 100), (1, ), (1, ))
end

@testset "Sampling structure" begin
    @test_throws ArgumentError SamplingSpace(TestShape(), rand(100) |> StateSpaceSet, rand(100) |> StateSpaceSet)
    @test_throws ArgumentError SamplingSpace(TestShape(), rand(2, 100), rand(2, 100))

    space = SamplingSpace(RectMicrostate(2), rand(100) |> StateSpaceSet, rand(100) |> StateSpaceSet)
    @test_throws ArgumentError RecurrenceMicrostatesAnalysis.get_sample(RecurrenceMicrostatesAnalysis.CPUCore(), TestSampling(), space)
    @test_throws ArgumentError RecurrenceMicrostatesAnalysis.get_num_samples(TestSampling(), space)
end

@testset "Shape structure" begin
    @test_throws ArgumentError RecurrenceMicrostatesAnalysis.compute_motif(TestShape(), ThresholdRecurrence(0.27), rand(2, 100), rand(2, 100), [1], [1], SVector{1, Int}(1))
    @test_throws ArgumentError RecurrenceMicrostatesAnalysis.get_histogram_size(TestShape())
    @test_throws ArgumentError RecurrenceMicrostatesAnalysis.get_power_vector(RecurrenceMicrostatesAnalysis.CPUCore(), TestShape())
    @test_throws ArgumentError RecurrenceMicrostatesAnalysis.get_offsets(RecurrenceMicrostatesAnalysis.CPUCore(), TestShape())
end