using Test
using Distances
using Distributions
using KernelAbstractions
using StaticArrays
using Random
using RecurrenceMicrostatesAnalysis

##  Triangle not implemented for spatial data:
@test_throws ArgumentError probabilities(RecurrenceMicrostates(0.27, TriangleMicrostate(3)), rand(1, 10, 10))

@testset "time series" begin
    x = rand(Uniform(0, 1), 1000) |> StateSpaceSet
    y = rand(Uniform(0, 1), 1000) |> StateSpaceSet
    rmspace = RecurrenceMicrostates(0.27, TriangleMicrostate(3); sampling_ratio = 0.1)

    @test sqrt(js_divergence(probabilities(rmspace, x), probabilities(rmspace, x))) / log(2) ≤ 0.1
    @test sqrt(js_divergence(probabilities(rmspace, x, y), probabilities(rmspace, x, y))) / log(2) ≤ 0.1
end

@testset "GPU" begin
    shape = TriangleMicrostate(3)
    core = RecurrenceMicrostatesAnalysis.GPUCore()

    @test RecurrenceMicrostatesAnalysis.get_power_vector(core, shape) isa SVector{6, Int32}
    @test RecurrenceMicrostatesAnalysis.get_offsets(core, shape) isa SVector{6, SVector{2, Int32}}
end