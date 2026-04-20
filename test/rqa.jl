using Test
using Distributions
using RecurrenceAnalysis
using RecurrenceMicrostatesAnalysis

@testset "Determinism" begin
    ##  We use a tolerance of 10% here.
    x = StateSpaceSet(rand(Uniform(0, 1), 1000))
    rp = RecurrenceMatrix(x, 0.27)
    det_l2 = determinism(rp)
    det_st = complexity(RecurrenceDeterminism(0.27), x)

    @test det_st isa Real
    @test (abs(det_l2 - det_st) / det_l2) ≤ 0.1


    rms_square = RecurrenceMicrostates(0.27, 3)
    rms_diagonal = RecurrenceMicrostates(0.27, DiagonalMicrostate(3))

    dist_square = probabilities(rms_square, x)
    dist_diagonal = probabilities(rms_diagonal, x)

    @test (abs(det_l2 - RecurrenceMicrostatesAnalysis.measure(RecurrenceDeterminism(0.27), rms_square, dist_square)) / det_l2) ≤ 0.1
    @test (abs(det_l2 - RecurrenceMicrostatesAnalysis.measure(RecurrenceDeterminism(0.27), rms_diagonal, dist_diagonal)) / det_l2) ≤ 0.1
end

@testset "Laminarity" begin
    ##  We use a tolerance of 10% here.
    x = StateSpaceSet(rand(Uniform(0, 1), 1000))
    rp = RecurrenceMatrix(x, 0.27)
    lam_l2 = laminarity(rp)
    lam_st = complexity(RecurrenceLaminarity(0.27), x)

    @test lam_st isa Real
    @test (abs(lam_l2 - lam_st) / lam_l2) ≤ 0.1


    rms_square = RecurrenceMicrostates(0.27, 3)
    rms_line = RecurrenceMicrostates(0.27, RectMicrostate(1, 3))

    dist_square = probabilities(rms_square, x)
    dist_line = probabilities(rms_line, x)

    @test (abs(lam_l2 - RecurrenceMicrostatesAnalysis.measure(RecurrenceLaminarity(0.27), rms_square, dist_square)) / lam_l2) ≤ 0.1
    @test (abs(lam_l2 - RecurrenceMicrostatesAnalysis.measure(RecurrenceLaminarity(0.27), rms_line, dist_line)) / lam_l2) ≤ 0.1
end

@testset "Recurrence rate" begin
    x = StateSpaceSet(rand(1000))
    dist = probabilities(RecurrenceMicrostates(0.27, 4), x)

    @test 0 ≤ RecurrenceMicrostatesAnalysis.measure(RecurrenceRate(0.27), dist) ≤ 1
    @test 0 ≤ complexity(RecurrenceRate(0.27), x) ≤ 1
end

@testset "Disorder" begin
    @test_throws AssertionError Disorder(1)
    @test_throws AssertionError Disorder(6)

    @testset "norm factor" begin
        @test RecurrenceMicrostatesAnalysis._norm_factor(Val(2), Val(1)) == 4
        @test RecurrenceMicrostatesAnalysis._norm_factor(Val(2), Val(2)) == 4
        @test RecurrenceMicrostatesAnalysis._norm_factor(Val(3), Val(1)) == 23
        @test RecurrenceMicrostatesAnalysis._norm_factor(Val(3), Val(2)) == 24
        @test RecurrenceMicrostatesAnalysis._norm_factor(Val(4), Val(1)) == 145
        @test RecurrenceMicrostatesAnalysis._norm_factor(Val(4), Val(2)) == 190
        @test RecurrenceMicrostatesAnalysis._norm_factor(Val(5), Val(1)) == 1173
        @test_throws ArgumentError RecurrenceMicrostatesAnalysis._norm_factor(Val(5), Val(2))
    end

    @testset "basic" begin
        x = StateSpaceSet(rand(Uniform(0, 1), 100))
        @test 0 ≤ complexity(Disorder(2), x) ≤ 1
        @test 0 ≤ complexity(Disorder(3), x) ≤ 1
        @test 0 ≤ complexity(Disorder(4), x) ≤ 1
        # @test 0 ≤ measure(Disorder(5), x) ≤ 1
    end
end

@testset "Windowed disorder" begin
    x = rand(Uniform(0, 1), 250)
    @test complexity(WindowedDisorder(20, 2), x) isa Vector{<:Real}
end