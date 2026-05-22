export SRandom

##########################################################################################
#   Sampling Mode: SRandom
##########################################################################################
"""
    SRandom{F<:Real} <: SamplingMode

Sampling mode that randomly selects microstate positions \$(i, j)\$ within the
sampling space.

#   Constructors
```julia
SRandom(num_samples::Int)
SRandom(ratio::Union{Float32, Float64})
```

The sampling mode can be initialized either by specifying the exact number of microstates
to sample or by providing a ratio of the total number of possible microstates.

#   Examples
```julia
s = SRandom(1000)   # Specify the exact number of sampled microstates
s = SRandom(0.05)   # Specify a ratio of the total possible microstates
```
"""
struct SRandom{F <: Real} <: SamplingMode
    sampling_factor::F
end
#.........................................................................................
function SRandom(num_samples::Int)
    @assert num_samples ≥ 1 throw(ArgumentError("number of samples must be greater than 1."))
    return SRandom{Int}(num_samples)
end 
#.........................................................................................
function SRandom(ratio::Union{Float32, Float64})
    @assert ratio > 0 throw(ArgumentError("sampling ratio must be greater than 0."))
    @assert ratio ≤ 1.0 throw(ArgumentError("sampling ratio must be smaller than 1."))

    return SRandom{typeof(ratio)}(ratio)
end

##########################################################################################
#   Implementation: sampling
##########################################################################################
#   Based on time series: (CPU)
#.........................................................................................
function get_sample(::CPUCore, ::SRandom, space::SSRect2, rng, _)
    i = rand(rng, 1:space.W)
    j = rand(rng, 1:space.H)

    return i, j
end

function get_sample(::CPUCore, ::SRandom, space::SSTriangle, rng, _)
    a = ((space.L - 2 * space.N + 1) * (space.L - 2 * space.N + 2)) ÷ 2
    m = rand(rng, 1:a)

    S = space.L - 2 * space.N + 1

    i = ceil(Int, ((2S + 1) - sqrt((2S + 1)^2 - 8m)) / 2)
    A = (i - 1) * S - ((i - 1) * (i - 2)) ÷ 2
    d = m - A - 1
    j = i + space.N + d

    return i, j
end

function get_sample(::CPUCore, ::SRandom, space::SSColumn, rng, _)
    i = space.I
    j = rand(rng, 1:space.H)

    return i, j
end
#.........................................................................................
#   Based on time series: (GPU)
#.........................................................................................
function get_sample(::GPUCore, ::SRandom, space::SSRect2, samples)
    return [@SVector[Int32(rand(1:space.W)), Int32(rand(1:space.H))] for _ in 1:samples]
end

function get_sample(::GPUCore, ::SRandom, space::SSTriangle, samples)
    throw("Triangle sampling space is not implemented for GPU when using random sampling mode.")
end

function get_sample(::GPUCore, ::SRandom, space::SSColumn, samples)
    return [@SVector[Int32(space.I), Int32(rand(1:space.H))] for _ in 1:samples]
end
#.........................................................................................
#   Based on spatial data: (CPU only)
#.........................................................................................
function get_sample(::CPUCore, ::SRandom, space::SSRectN, idx::Vector{Int}, rng, _)
    for i in eachindex(idx)
        idx[i] = rand(rng, 1:space.space[i])
    end
end
##########################################################################################
#   Implementations: Utils
##########################################################################################
get_num_samples(mode::SRandom{<:Integer}, ::SSRect2) = mode.sampling_factor
get_num_samples(mode::SRandom{<:Integer}, ::SSRectN) = mode.sampling_factor
get_num_samples(mode::SRandom{<:Integer}, ::SSTriangle) = mode.sampling_factor
get_num_samples(mode::SRandom{<:Integer}, ::SSColumn) = mode.sampling_factor
#.........................................................................................
get_num_samples(mode::SRandom{<:Real}, space::SSRect2) = ceil(Int, mode.sampling_factor * space.W * space.H)
get_num_samples(mode::SRandom{<:Real}, space::SSRectN) = ceil(Int, mode.sampling_factor * reduce(*, space.space))
get_num_samples(mode::SRandom{<:Real}, space::SSTriangle) = ceil(Int, mode.sampling_factor * ((space.L - 2 * space.N + 1) * (space.L - 2 * space.N + 2)) ÷ 2)
get_num_samples(mode::SRandom{<:Real}, space::SSColumn) = ceil(Int, mode.sampling_factor * space.H)
##########################################################################################