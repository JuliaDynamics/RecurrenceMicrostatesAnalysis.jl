export Full

##########################################################################################
#   Sampling Mode: Random
##########################################################################################
"""
    Full <: SamplingMode

Sampling mode that selects **all** possible microstates with in the
sampling space.

#   Constructor
```julia
s = Full()
```

!!! warning
    The **Full** sampling mode is not supported for spatial data.
"""
struct Full <: SamplingMode end

##########################################################################################
#   Implementation: sampling
##########################################################################################
#   Based on time series: (CPU)
#.........................................................................................
function get_sample(::CPUCore, ::Full, space::SSRect2, _, m)
    i = ((m - 1) % space.W) + 1
    j = ((m - 1) ÷ space.W) + 1

    return i, j
end

function get_sample(::CPUCore, ::Full, space::SSTriangle, _, m)
    S = space.L - 2 * space.N + 1

    i = ceil(Int, ((2S + 1) - sqrt((2S + 1)^2 - 8m)) / 2)
    A = (i - 1) * S - ((i - 1) * (i - 2)) ÷ 2
    d = m - A - 1
    j = i + space.N + d

    return i, j
end

function get_sample(::CPUCore, ::Full, space::SSColumn, _, m)
    i = space.I
    j = m

    return i, j
end
#.........................................................................................
#   Based on time series: (GPU)
#.........................................................................................
function get_sample(::GPUCore, ::Full, space::SSRect2, _, m)
    i = Int32((m - 1) % space.W) + 1
    j = Int32((m - 1) ÷ space.W) + 1

    return i, j
end

function get_sample(::GPUCore, ::Full, space::SSTriangle, _, m)
    throw("Triangle sampling space is not implemented for GPU when using full sampling mode.")
end

function get_sample(::GPUCore, ::Full, space::SSColumn, _, m)
    i = space.I
    j = m

    return i, j
end

##########################################################################################
#   Implementations: Utils
##########################################################################################
get_num_samples(::Full, space::SSRect2) = space.W * space.H
get_num_samples(::Full, space::SSTriangle) = ((space.L - 2 * space.N + 1) * (space.L - 2 * space.N + 2)) ÷ 2
get_num_samples(::Full, space::SSColumn) = space.H

##########################################################################################