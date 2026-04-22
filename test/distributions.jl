using Test
using ComplexityMeasures
using RecurrenceMicrostatesAnalysis

@testset "time series" begin
    x = rand(100) |> StateSpaceSet
    y = rand(200) |> StateSpaceSet

    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, 3), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, 3; ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, 3; sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, 3; metric = Cityblock()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, 3; ratio = 0.1, metric = Cityblock()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, 3; sampling = Full(), metric = Cityblock()), x)) - 1) <= 1e-3

    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), 3; sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), 3; ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), 3; sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), 3; ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), 3; sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), 3; ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), 3; sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), 3; ratio = 0.1), x)) - 1) <= 1e-3

    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), RectMicrostate(3); sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), RectMicrostate(3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), RectMicrostate(3); sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), RectMicrostate(3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), RectMicrostate(3); sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), RectMicrostate(3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), RectMicrostate(3); sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), RectMicrostate(3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), RectMicrostate(2, 3); sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), RectMicrostate(2, 3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), RectMicrostate(2, 3); sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), RectMicrostate(2, 3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), RectMicrostate(2, 3); sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), RectMicrostate(2, 3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), RectMicrostate(2, 3); sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), RectMicrostate(2, 3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), DiagonalMicrostate(3); sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), DiagonalMicrostate(3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), DiagonalMicrostate(3); sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), DiagonalMicrostate(3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), DiagonalMicrostate(3); sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), DiagonalMicrostate(3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), DiagonalMicrostate(3); sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), DiagonalMicrostate(3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), TriangleMicrostate(3); sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), TriangleMicrostate(3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), TriangleMicrostate(3); sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), TriangleMicrostate(3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), TriangleMicrostate(3); sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), TriangleMicrostate(3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), TriangleMicrostate(3); sampling = Full()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), TriangleMicrostate(3); ratio = 0.1), x)) - 1) <= 1e-3

    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, 3), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, 3; ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, 3; sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, 3; metric = Cityblock()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, 3; ratio = 0.1, metric = Cityblock()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, 3; sampling = Full(), metric = Cityblock()), x, y)) - 1) <= 1e-3

    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), 3; sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), 3; ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), 3; sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), 3; ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), 3; sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), 3; ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), 3; sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), 3; ratio = 0.1), x, y)) - 1) <= 1e-3

    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), RectMicrostate(3); sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), RectMicrostate(3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), RectMicrostate(3); sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), RectMicrostate(3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), RectMicrostate(3); sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), RectMicrostate(3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), RectMicrostate(3); sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), RectMicrostate(3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), RectMicrostate(2, 3); sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), RectMicrostate(2, 3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), RectMicrostate(2, 3); sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), RectMicrostate(2, 3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), RectMicrostate(2, 3); sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), RectMicrostate(2, 3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), RectMicrostate(2, 3); sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), RectMicrostate(2, 3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), DiagonalMicrostate(3); sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), DiagonalMicrostate(3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), DiagonalMicrostate(3); sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), DiagonalMicrostate(3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), DiagonalMicrostate(3); sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), DiagonalMicrostate(3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), DiagonalMicrostate(3); sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), DiagonalMicrostate(3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), TriangleMicrostate(3); sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), TriangleMicrostate(3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), TriangleMicrostate(3); sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), TriangleMicrostate(3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), TriangleMicrostate(3); sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), TriangleMicrostate(3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), TriangleMicrostate(3); sampling = Full()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), TriangleMicrostate(3); ratio = 0.1), x, y)) - 1) <= 1e-3
end

@testset "spatial data" begin
    x = rand(1, 20, 20)
    y = rand(1, 10, 10)

    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, (2, 1, 2, 1)), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, (2, 1, 2, 1); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, (2, 1, 2, 1); metric = Cityblock()), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, (2, 1, 2, 1); ratio = 0.1, metric = Cityblock()), x)) - 1) <= 1e-3

    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), (2, 1, 2, 1); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), (2, 1, 2, 1); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), (2, 1, 2, 1); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), (2, 1, 2, 1); ratio = 0.1), x)) - 1) <= 1e-3

    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), RectMicrostate((2, 1, 2, 1)); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), RectMicrostate((2, 1, 2, 1)); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), RectMicrostate((2, 1, 2, 1)); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), RectMicrostate((2, 1, 2, 1)); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), DiagonalMicrostate(3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), DiagonalMicrostate(3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), DiagonalMicrostate(3); ratio = 0.1), x)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), DiagonalMicrostate(3); ratio = 0.1), x)) - 1) <= 1e-3

    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, (2, 1, 2, 1)), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, (2, 1, 2, 1); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, (2, 1, 2, 1); metric = Cityblock()), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(0.27, (2, 1, 2, 1); ratio = 0.1, metric = Cityblock()), x, y)) - 1) <= 1e-3

    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), (2, 1, 2, 1); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), (2, 1, 2, 1); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), (2, 1, 2, 1); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), (2, 1, 2, 1); ratio = 0.1), x, y)) - 1) <= 1e-3

    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), RectMicrostate((2, 1, 2, 1)); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), RectMicrostate((2, 1, 2, 1)); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), RectMicrostate((2, 1, 2, 1)); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), RectMicrostate((2, 1, 2, 1)); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27), DiagonalMicrostate(3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(ThresholdRecurrence(0.27; metric = Cityblock()), DiagonalMicrostate(3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27), DiagonalMicrostate(3); ratio = 0.1), x, y)) - 1) <= 1e-3
    @test abs(sum(probabilities(RecurrenceMicrostates(CorridorRecurrence(0.05, 0.27; metric = Cityblock()), DiagonalMicrostate(3); ratio = 0.1), x, y)) - 1) <= 1e-3
end