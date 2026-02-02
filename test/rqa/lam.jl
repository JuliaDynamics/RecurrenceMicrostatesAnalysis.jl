using Test
using Distributions
using RecurrenceAnalysis
using RecurrenceMicrostatesAnalysis

@testset "invalid distribution" begin
    rmspace = RecurrenceMicrostates(0.27, 2)
    dist = probabilities(rmspace, rand(100))
    @test_throws ArgumentError measure(Laminarity(), rmspace, dist)
end

##  We use a tolerance of 10% here.
@testset "estimation error" begin
    x = StateSpaceSet(rand(Uniform(0, 1), 1000))
    rp = RecurrenceMatrix(x, 0.27)
    det_l2 = laminarity(rp)

    @test measure(Laminarity(), x) isa Real
    @test (abs(det_l2 - measure(Laminarity(), x)) / det_l2) ≤ 0.1

    rms_square = RecurrenceMicrostates(0.27, 3)
    rms_line = RecurrenceMicrostates(0.27, RectMicrostate(1, 3))

    dist_square = probabilities(rms_square, x)
    dist_line = probabilities(rms_line, x)

    @test (abs(det_l2 - measure(Laminarity(), rms_square, dist_square)) / det_l2) ≤ 0.1
    @test (abs(det_l2 - measure(Laminarity(), rms_line, dist_line)) / det_l2) ≤ 0.1
end