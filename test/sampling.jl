using Test
using Distances
using Distributions
using KernelAbstractions
using StaticArrays
using Random
using RecurrenceMicrostatesAnalysis

@testset "Full" begin
    ##  Full not implemented for spatial data:
    @test_throws ArgumentError probabilities(RecurrenceMicrostates(0.27, (2, 1, 2, 1); sampling = Full()), rand(1, 10, 10))

    @testset "num samples" begin
        data = rand(100) |> StateSpaceSet
        space = RecurrenceMicrostatesAnalysis.SamplingSpace(RectMicrostate(2), RectSamplingSpace(), data, data)

        @test RecurrenceMicrostatesAnalysis.get_num_samples(Full(), space) == space.W * space.H
        @test RecurrenceMicrostatesAnalysis.get_num_samples(Full(), space) isa Integer
    end

    @testset "get sample" begin
        @testset "CPU" begin
            data_1 = rand(50) |> StateSpaceSet
            data_2 = rand(20) |> StateSpaceSet
            space = RecurrenceMicrostatesAnalysis.SamplingSpace(RectMicrostate(2), RectSamplingSpace(), data_1, data_2)
            core = RecurrenceMicrostatesAnalysis.CPUCore()

            @test RecurrenceMicrostatesAnalysis.get_sample(core, Full(), space, nothing, 10) isa Tuple{<: Integer, <: Integer}
            
            samples = RecurrenceMicrostatesAnalysis.get_num_samples(Full(), space)
            for m ∈ 1:samples
                i, j = RecurrenceMicrostatesAnalysis.get_sample(core, Full(), space, nothing, m)
                @test 1 ≤ i ≤ space.W
                @test 1 ≤ j ≤ space.H
            end
        end

        @testset "GPU" begin
            data_1 = rand(50) |> StateSpaceSet
            data_2 = rand(20) |> StateSpaceSet
            space = RecurrenceMicrostatesAnalysis.SamplingSpace(RectMicrostate(2), RectSamplingSpace(), data_1, data_2)
            core = RecurrenceMicrostatesAnalysis.GPUCore()

            @test RecurrenceMicrostatesAnalysis.get_sample(core, Full(), space, nothing, 10) isa Tuple{<: Integer, <: Integer}

            samples = RecurrenceMicrostatesAnalysis.get_num_samples(Full(), space)
            for m ∈ 1:samples
                i, j = RecurrenceMicrostatesAnalysis.get_sample(core, Full(), space, nothing, m)
                @test 1 ≤ i ≤ space.W
                @test 1 ≤ j ≤ space.H
            end
        end
    end

    @testset "values" begin
        x = StateSpaceSet(rand(100))
        y = StateSpaceSet(rand(50))
        rmspace = RecurrenceMicrostates(0.27, 3; sampling = Full())

        @test js_divergence(probabilities(rmspace, x), probabilities(rmspace, x)) == 0
        @test js_divergence(probabilities(rmspace, x, y), probabilities(rmspace, x, y)) == 0
    end
end

@testset "Random" begin
    @test_throws ArgumentError SRandom(0)
    @test_throws ArgumentError SRandom(0.0)
    @test_throws ArgumentError SRandom(2.0)

    @testset "num samples" begin
        @testset "time series" begin
            data = rand(100) |> StateSpaceSet
            space = RecurrenceMicrostatesAnalysis.SamplingSpace(RectMicrostate(2), RectSamplingSpace(), data, data)

            sampling = SRandom(0.05)
            @test RecurrenceMicrostatesAnalysis.get_num_samples(sampling, space) == ceil(Int, sampling.sampling_factor * space.W * space.H)
            @test RecurrenceMicrostatesAnalysis.get_num_samples(sampling, space) isa Integer

            sampling = SRandom(10)
            @test RecurrenceMicrostatesAnalysis.get_num_samples(sampling, space) == sampling.sampling_factor
            @test RecurrenceMicrostatesAnalysis.get_num_samples(sampling, space) isa Integer
        end
        
        @testset "spatial data" begin
            data = rand(2, 100)
            space = RecurrenceMicrostatesAnalysis.SamplingSpace(RectMicrostate((2, 2)), RectSamplingSpace(), data, data)

            sampling = SRandom(0.05)
            @test RecurrenceMicrostatesAnalysis.get_num_samples(sampling, space) == ceil(Int, sampling.sampling_factor * reduce(*, space.space))
            @test RecurrenceMicrostatesAnalysis.get_num_samples(sampling, space) isa Integer

            sampling = SRandom(10)
            @test RecurrenceMicrostatesAnalysis.get_num_samples(sampling, space) == sampling.sampling_factor
            @test RecurrenceMicrostatesAnalysis.get_num_samples(sampling, space) isa Integer
        end
    end

    @testset "get sample" begin
        @testset "CPU" begin
            data_1 = rand(50) |> StateSpaceSet
            data_2 = rand(20) |> StateSpaceSet
            space = RecurrenceMicrostatesAnalysis.SamplingSpace(RectMicrostate(2), RectSamplingSpace(), data_1, data_2)
            sampling = SRandom(50)
            core = RecurrenceMicrostatesAnalysis.CPUCore()

            @test RecurrenceMicrostatesAnalysis.get_sample(core, sampling, space, TaskLocalRNG(), nothing) isa Tuple{<: Integer, <: Integer}
            
            samples = RecurrenceMicrostatesAnalysis.get_num_samples(SRandom(0.05), space)
            for m ∈ 1:samples
                i, j = RecurrenceMicrostatesAnalysis.get_sample(core, sampling, space, TaskLocalRNG(), nothing)
                @test 1 ≤ i ≤ space.W
                @test 1 ≤ j ≤ space.H
            end
        end

        @testset "GPU" begin
            data_1 = rand(50) |> StateSpaceSet
            data_2 = rand(20) |> StateSpaceSet
            space = RecurrenceMicrostatesAnalysis.SamplingSpace(RectMicrostate(2), RectSamplingSpace(), data_1, data_2)
            sampling = SRandom(50)
            core = RecurrenceMicrostatesAnalysis.GPUCore()

            @test RecurrenceMicrostatesAnalysis.get_sample(core, sampling, space, 2) isa Vector{SVector{2, Int32}}

            num_samples = RecurrenceMicrostatesAnalysis.get_num_samples(SRandom(0.05), space)
            samples = RecurrenceMicrostatesAnalysis.get_sample(core, sampling, space, num_samples)
            for m ∈ samples
                i, j = m
                @test 1 ≤ i ≤ space.W
                @test 1 ≤ j ≤ space.H
            end
        end

        @testset "spatial data" begin
            data_1 = rand(2, 50, 50)
            data_2 = rand(2, 20, 20)
            space = RecurrenceMicrostatesAnalysis.SamplingSpace(RectMicrostate((2, 1, 1, 2)), RectSamplingSpace(), data_1, data_2)
            sampling = SRandom(50)
            core = RecurrenceMicrostatesAnalysis.CPUCore()

            idx = zeros(Int, 4)
            @test RecurrenceMicrostatesAnalysis.get_sample(core, sampling, space, idx, TaskLocalRNG(), nothing) isa Nothing
            
            samples = RecurrenceMicrostatesAnalysis.get_num_samples(SRandom(0.05), space)
            for m ∈ 1:samples
                RecurrenceMicrostatesAnalysis.get_sample(core, sampling, space, idx, TaskLocalRNG(), nothing)
                for i ∈ eachindex(idx)
                    @test 1 ≤ idx[i] ≤ space.space[i]
                end
            end
        end
    end

    @testset "values" begin

        @testset "time series" begin
            x = rand(Uniform(0, 1), 1000) |> StateSpaceSet
            y = rand(Uniform(0, 1), 1000) |> StateSpaceSet
            rmspace = RecurrenceMicrostates(0.27, 3; ratio = 0.1)

            @test sqrt(js_divergence(probabilities(rmspace, x), probabilities(rmspace, x))) / log(2) ≤ 0.1
            @test sqrt(js_divergence(probabilities(rmspace, x, y), probabilities(rmspace, x, y))) / log(2) ≤ 0.1
        end

        @testset "spatial data" begin
            x = rand(Uniform(0, 1), (1, 50, 50))
            y = rand(Uniform(0, 1), (1, 50, 50))
            rmspace = RecurrenceMicrostates(0.27, (2, 1, 2, 1); ratio = 0.1)

            @test sqrt(js_divergence(probabilities(rmspace, x), probabilities(rmspace, x))) / log(2) ≤ 0.1
            @test sqrt(js_divergence(probabilities(rmspace, x, y), probabilities(rmspace, x, y))) / log(2) ≤ 0.1
        end
        
    end
end