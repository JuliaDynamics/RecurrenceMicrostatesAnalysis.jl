using Test
using RecurrenceMicrostatesAnalysis

@testset "Threshold recurrence" begin
    @test_throws ArgumentError ThresholdRecurrence(-1.0)

    @testset "time series" begin
        x = rand(100) |> StateSpaceSet
        y = rand(100) |> StateSpaceSet
        
        standard = ThresholdRecurrence(0.27)

        @test RecurrenceMicrostatesAnalysis.recurrence(standard, x, y, 1, 1) isa UInt8

        r = RecurrenceMicrostatesAnalysis.recurrence(standard, x, y, 1, 1)
        @test r == UInt8(0) || r == UInt8(1)
    end

    @testset "GPU (running on CPU)" begin
        x = rand(100) |> StateSpaceSet
        y = rand(100) |> StateSpaceSet
        
        standard = ThresholdRecurrence(0.27; metric = GPUEuclidean())

        @test RecurrenceMicrostatesAnalysis.gpu_recurrence(standard, x, y, 1, 1, 1) isa UInt8

        r = RecurrenceMicrostatesAnalysis.gpu_recurrence(standard, x, y, 1, 1, 1)
        @test r == UInt8(0) || r == UInt8(1)
    end

    @testset "spatial data" begin
        x = rand(2, 100)
        y = rand(2, 100)
        
        standard = ThresholdRecurrence(0.27)

        pos = (1, )
        @test RecurrenceMicrostatesAnalysis.recurrence(standard, x, y, pos, pos) isa UInt8

        r = RecurrenceMicrostatesAnalysis.recurrence(standard, x, y, pos, pos)
        @test r == UInt8(0) || r == UInt8(1)
    end
end

@testset "Corridor recurrence" begin
    @test_throws ArgumentError CorridorRecurrence(1.0, 0.0)
    @test_throws ArgumentError CorridorRecurrence(-1.0, 0.27)

    @testset "time series" begin
        x = rand(100) |> StateSpaceSet
        y = rand(100) |> StateSpaceSet
        
        corridor = CorridorRecurrence(0.05, 0.27)

        @test RecurrenceMicrostatesAnalysis.recurrence(corridor, x, y, 1, 1) isa UInt8

        r = RecurrenceMicrostatesAnalysis.recurrence(corridor, x, y, 1, 1)
        @test r == UInt8(0) || r == UInt8(1)

        ##  it must be zero when computing for the same position.
        @test RecurrenceMicrostatesAnalysis.recurrence(corridor, x, x, 1, 1) == UInt8(0)
    end

    @testset "GPU (running on CPU)" begin
        x = rand(100) |> StateSpaceSet
        y = rand(100) |> StateSpaceSet
        
        corridor = CorridorRecurrence(0.05, 0.27; metric = GPUEuclidean())

        @test RecurrenceMicrostatesAnalysis.gpu_recurrence(corridor, x, y, 1, 1, 1) isa UInt8

        r = RecurrenceMicrostatesAnalysis.gpu_recurrence(corridor, x, y, 1, 1, 1)
        @test r == UInt8(0) || r == UInt8(1)

        ##  it must be zero when computing for the same position.
        @test RecurrenceMicrostatesAnalysis.gpu_recurrence(corridor, x, x, 1, 1, 1) == UInt8(0)
    end

    @testset "spatial data" begin
        x = rand(2, 100)
        y = rand(2, 100)
        
        corridor = CorridorRecurrence(0.05, 0.27)

        pos = (1, )
        @test RecurrenceMicrostatesAnalysis.recurrence(corridor, x, y, pos, pos) isa UInt8

        r = RecurrenceMicrostatesAnalysis.recurrence(corridor, x, y, pos, pos)
        @test r == UInt8(0) || r == UInt8(1)

        ##  it must be zero when computing for the same position.
        @test RecurrenceMicrostatesAnalysis.recurrence(corridor, x, x, pos, pos) == UInt8(0)
    end
end