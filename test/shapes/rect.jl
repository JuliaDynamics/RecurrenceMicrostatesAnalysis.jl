using Test
using Distances
using Distributions
using KernelAbstractions
using StaticArrays
using Random
using RecurrenceMicrostatesAnalysis

@testset "time series" begin
    x = rand(Uniform(0, 1), 1000) |> StateSpaceSet
    y = rand(Uniform(0, 1), 1000) |> StateSpaceSet
    rmspace = RecurrenceMicrostates(0.27, 3; sampling_ratio = 0.1)

    @test sqrt(js_divergence(probabilities(rmspace, x), probabilities(rmspace, x))) / log(2) ≤ 0.1
    @test sqrt(js_divergence(probabilities(rmspace, x, y), probabilities(rmspace, x, y))) / log(2) ≤ 0.1
end

@testset "spatial data" begin
    x = rand(Uniform(0, 1), (1, 50, 50))
    y = rand(Uniform(0, 1), (1, 50, 50))
    rmspace = RecurrenceMicrostates(0.27, (2, 1, 2, 1); sampling_ratio = 0.1)

    @test sqrt(js_divergence(probabilities(rmspace, x), probabilities(rmspace, x))) / log(2) ≤ 0.1
    @test sqrt(js_divergence(probabilities(rmspace, x, y), probabilities(rmspace, x, y))) / log(2) ≤ 0.1
end

@testset "GPU" begin
    shape = RectMicrostate(3)
    core = RecurrenceMicrostatesAnalysis.GPUCore()

    @test RecurrenceMicrostatesAnalysis.get_power_vector(core, shape) isa SVector{9, Int32}
    @test RecurrenceMicrostatesAnalysis.get_offsets(core, shape) isa SVector{9, SVector{2, Int32}}
end