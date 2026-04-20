using Test
using Distributions
using RecurrenceMicrostatesAnalysis

@testset "Operations" begin
    @test operate(PermuteColumns(RectMicrostate(3, 3)), 237, 2) == 347
    @test operate(PermuteRows(RectMicrostate(3, 3)), 237, [1, 3, 2]) == 349
    @test operate(Transpose(RectMicrostate(3, 3)), 237) == 231
end

@testset "Optimization" begin
    data = rand(Uniform(0, 1), 1000) |> StateSpaceSet
    @test (optimize(Threshold(), Shannon(), 2, data)[1] - 0.27) / 0.27 ≤ 0.15
    @test (optimize(Threshold(), Disorder(4), data)[1] - 0.27) / 0.27 ≤ 0.15
    @test optimize(Threshold(), Shannon(), 2, data) isa Tuple{Float64, Float64}
    @test optimize(Threshold(), Disorder(4), data) isa Tuple{Float64, Float64}

    data = rand(Uniform(0, 1), 1001) |> StateSpaceSet
    @test (optimize(Threshold(), Shannon(), 2, data)[1] - 0.27) / 0.27 ≤ 0.15
    @test (optimize(Threshold(), Disorder(4), data)[1] - 0.27) / 0.27 ≤ 0.15
    @test optimize(Threshold(), Shannon(), 2, data) isa Tuple{Float64, Float64}
    @test optimize(Threshold(), Disorder(4), data) isa Tuple{Float64, Float64}
end

@testset "GPU Metrics" begin
    data = rand(10) |> StateSpaceSet
    @test RecurrenceMicrostatesAnalysis.gpu_evaluate(GPUEuclidean(), data, data, 1, 1, 1) == 0
end

