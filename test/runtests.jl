using Test
using Aqua
using ComplexityMeasures
using RecurrenceMicrostatesAnalysis

defaultname(file) = uppercasefirst(replace(splitext(basename(file))[1], '_' => ' '))
testfile(file, testname = defaultname(file)) = @testset "$testname" begin; include(file); end

# Aqua testing
@testset "Check using Aqua.jl" begin
    Aqua.test_all(RecurrenceMicrostatesAnalysis)
end

@testset "RecurrenceMicrostatesAnalysis.jl" begin
    testfile("core.jl")
    testfile("recurrences.jl")
    testfile("sampling.jl")
    testfile("shapes.jl")
    testfile("rqa.jl")
    testfile("utils.jl")

    testfile("distributions.jl")
end