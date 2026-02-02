using Test
using Distributions
using RecurrenceAnalysis
using RecurrenceMicrostatesAnalysis

@testset "invalid distribution" begin
    rmspace = RecurrenceMicrostates(0.27, 2)
    dist = probabilities(rmspace, rand(Uniform(0, 1), 100))
    @test_throws ArgumentError measure(Determinism(), rmspace, dist)
end

##  We use a tolerance of 10% here.
@testset "estimation error" begin
    x = StateSpaceSet(rand(Uniform(0, 1), 1000))
    rp = RecurrenceMatrix(x, 0.27)
    det_l2 = determinism(rp)

    @test measure(Determinism(), x) isa Real
    @test (abs(det_l2 - measure(Determinism(), x)) / det_l2) ≤ 0.1


    rms_square = RecurrenceMicrostates(0.27, 3)
    rms_diagonal = RecurrenceMicrostates(0.27, DiagonalMicrostate(3))

    dist_square = probabilities(rms_square, x)
    dist_diagonal = probabilities(rms_diagonal, x)

    @test (abs(det_l2 - measure(Determinism(), rms_square, dist_square)) / det_l2) ≤ 0.1
    @test (abs(det_l2 - measure(Determinism(), rms_diagonal, dist_diagonal)) / det_l2) ≤ 0.1
end