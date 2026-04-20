# # [Tutorial for RecurrenceMicrostatesAnalysis.jl](@id tutorial)

# In this tutorial we go through a typical usage of  **RecurrenceMicrostatesAnalysis.jl**.
# We'll see how to calculate distributions of recurrence microstates,
# how to optimize our choices regarding the distribution generation,
# and how to perform Recurrence Microstate Analysis (**RMA**).

# !!! info "ComplexityMeasures.jl"
#     RecurrenceMicrostatesAnalysis.jl interfaces with, and extends, ComplexityMeasures.jl.
#     It can enhance your understanding if you have first view the [tutorial of ComplexityMeasures.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/tutorial/). 
#     Regardless the current tutorial is written to be self-contained.

# ## Crash-course into RMA

# Consider a time series $\vec{x}_i \in \mathbb{R}^d$, $i \in \{1, 2, \dots, K\}$,
# where $K$ is the length of the time series and $d$ is the dimension of the phase space.
# The recurrence matrix
# ```math
# R_{i,j} = \Theta(\varepsilon - \|\vec x_i - \vec x_j\|),
# ```
# where $\Theta(\cdot)$ denotes the Heaviside step function and $\varepsilon$ is the recurrence
# threshold.

# The following figure shows examples of recurrence matrices for different systems:
# (a) white noise;
# (b) a superposition of harmonic oscillators;
# (c) a logistic map with a linear trend;
# (d) Brownian motion.

# ![Image of four RPs with their timeseries](assets/rps.png)

# A recurrence microstate is a local structure extracted from a recurrence matrix. For a given microstate
# shape and size, the set of possible microstates is finite. For example, square microstates
# with size $N = 2$ yield $16$ distinct configurations.

# ![Image of the 16 squared microstates to N = 2](assets/microstates.png)

# Recurrence Microstates Analysis (RMA) uses the probability distribution of these microstates
# as a source of information for characterizing dynamical systems.

# ## Probability distributions of recurrence microstates

# Extracting propabilities corresponding to recurrence microstates
# is done via the ComplexityMeasures.jl interface and it is relatively straightforward.
# We first specify the options of [`RecurrenceMicrostates`](@ref),
# which essentially means e.g., what sort of distance threshold defines a recurrence,
# and what is maximum microstate size to consider. Then we pass this to functions
# like `probabilities`, `entropy`, etc.

# Let's first generate some data of a chaotic map using **DynamicalSystems.jl**:

using DynamicalSystemsBase

function henon_rule(u, p, n)
    x, y = u # system state
    a, b = p # system parameters
    xn = 1.0 - a*x^2 + y
    yn = b*x
    return SVector(xn, yn)
end

u0 = [0.2, 0.3]
p0 = [1.4, 0.3]
henon = DeterministicIteratedMap(henon_rule, u0, p0)
X, t = trajectory(henon, 10_000)
X

# Notice that `X` is already a [`StateSpaceSet`](@ref). Because  **RecurrenceMicrostatesAnalysis.jl**
# is part of **DynamicalSystems.jl**, this data type is the preferred input type.
# Other types are also possible as we described in the [API](@ref).

# Now, we specify the recurrence microstate configuration

using RecurrenceMicrostatesAnalysis

ε = 0.25
N = 2
rmspace = RecurrenceMicrostates(ε, N)

# and finally call

probs = probabilities(rmspace, X)

# The `probabilities` function is the same function as in **ComplexityMeasures.jl**.
# Given an outcome space, that is a way to _symbolize_ input data into discrete outcomes,
# `probabilities` return the probability (relative occurrence frequency) for each outcome.
# And indeed, the recurrence microstates is an outcome space.

# If instead of the probabilities of the microstates you want their direct count
# simply replace `probabilities` with `counts`.

counts(rmspace, X)

# ## Recurrence microstates analysis (RMA)
# To actually analyze your data, there are two ways forwards.
# One way, is to utilize these probabilities within the interface provided
# by **ComplexityMeasures.jl** to calculate entropies.
# For example, the corresponding Shannon entropy is

entropy(Shannon(), probs)

# (note that the API of `ComplexityMeasures` is re-exported by `RecurrenceMicrostateAnalysis`).
# This number corresponds to the **recurrence microstate entropy** as defined in our
# publication [Corso2018Entropy](@cite).

# `ComplexityMeasures` allows the convenience syntax of

entropy(Shannon(), rmspace, X)

# so in this case we wouldn't need to calculate the probabilities directly.
# Naturally, any other entropy could be estimated instead,

entropy(Tsallis(), rmspace, X)

# although we haven't explored alternative entropies in research yet.

# The second way forward is the more traditional recurrence quantification analysis
# route, where you estimate (approximately, really) various quantities
# such as laminarity that fundamentally relate to the context of recurrences.

# These quantities are estimated using a `ComplexityEstimator`, similar to
# **ComplexityMeasures.jl**. We begin by defining our estimators:

# - For recurrence rate:

rr_estimator = RecurrenceRate(ε)

# - For laminarity:

lam_estimator = RecurrenceLaminarity(ε)

# - For determinism:

det_estimator = RecurrenceDeterminism(ε)

# Then, we use the `complexity` function to estimate the quantities:

rr = complexity(rr_estimator, X)
lam = complexity(lam_estimator, X)
det = complexity(det_estimator, X)

rr, lam, det

# We can compare these values with the exact values computed using **RecurrenceAnalysis.jl**.

using RecurrenceAnalysis

rp = RecurrenceMatrix(X, ε)
qt = rqa(rp)

qt[:RR], qt[:LAM], qt[:DET]

# All of these quantities, like laminarity, are in fact _complexity measures_,
# which is why **RecurrenceMicrostateAnalysis.jl** fits so well within the
# interface of **ComplexityMeasures.jl**.

# We have also implemented a unified function to compute all RMA estimations,
# similar to the `rqa` function from **RecurrenceAnalysis.jl**. This is the
# `rma` function:

rma(ε, X)

# ## Disorder

# Recurrence Microstates Analysis also introduces a novel quantifier:
# "Disorder index via symmetry in recurrence microstates" (DISREM). This quantifier
# uses the equiprobability property of recurrence microstates, due to the disorder
# condition, to quantify the disorder of a sequence of data elements. We also estimate
# disorder using a complexity estimator:

disorder = Disorder()

# Then, we can estimate disorder for our time series:

complexity(disorder, X)

# Note that disorder is free of parameters (except for the microstate length and
# the number of thresholds used). This is because the quantifier is
# defined using the maximum total entropy of recurrence microstate classes, considering
# a large range of thresholds.

# It is also possible to estimate disorder while splitting the data into windows.
# We prepared a special complexity estimator for this, aiming to facilitate
# its usage. In this situation, you must define a [`WindowedDisorder`](@ref).

window_len = 1000
win_disorder = WindowedDisorder(window_len; step = 100)

# Here, we are using windows of length 1000 points, moved in steps of 100 points.
# Finally, we compute the quantifier for each window:

wd = complexity(win_disorder, X)

# Plotting it:
using CairoMakie
lines(wd)

# !!! compat "Disorder specifications"
#     Disorder is only available for the standard outcome space extracted
#     from an RP using square microstates. Therefore, it is not compatible
#     with different microstate shapes or input variations such as CRP
#     and SRP, which will be introduced later.

# !!! info "Performance"
#     Disorder uses several recurrence microstate distributions computed
#     using a full sampling mode. It is a heavy quantifier that can have
#     a high computational cost. For large time series, we recommend
#     computing this quantifier using [GPU](@ref).

# ## Optimization of free parameters

# It is known that recurrence analysis has some free parameters. The most
# important of them is the recurrence threshold, $\varepsilon$, used to
# define which elements from a sequence are recurrent and which are not.
# In practice, these free parameters are not a major issue and allow
# different analyses of the same system. Of course, **RecurrenceMicrostateAnalysis.jl**
# considers this and allows you to use any value as your threshold.

# Nevertheless, RMA has two situations where it is necessary to set
# a specific recurrence threshold: when computing disorder and when using
# RMA distributions as input for machine learning.

# In both of these situations, it is important to use a threshold that
# maximizes a certain quantity. Disorder is defined as the maximum total
# entropy per class; therefore, the recurrence threshold is not truly a free
# parameter in this case, since there is an optimal threshold that results
# in the maximum observable disorder. Moreover, when working with RMA and
# machine learning (see [this example](@ref example_ml)),
# in most cases there is a correlation between the accuracy and the distribution
# that maximizes the recurrence entropy [Spezzatto2024ML](@cite). Thus, it is a good 
# idea to use this as a basis for defining an optimal value for the recurrence threshold.

# With this in mind, we provide a function to optimize some [`Parameter`](@ref)
# based on a given quantity. Currently, the only available parameter is
# [`Threshold`](@ref), which can be optimized using the function [`optimize`](@ref)
# by maximizing recurrence entropy or disorder.

# Using recurrence entropy as an example:
ε, S = optimize(Threshold(), Shannon(), N, X)
rmspace = RecurrenceMicrostates(ε, N)
h = entropy(Shannon(), rmspace, X)
(h, S)

# Or for disorder:
ε, ξ = optimize(Threshold(), disorder, X)
Ξ = complexity(disorder, X)

(Ξ, ξ)

# Note that there is a difference between `Ξ` and `ξ`. The `optimize` function uses
# a sampling ratio of $10\%$, which can result in a considerably different value.
# Internally, this optimization structure is used to define an optimal threshold range,
# from which we extract the disorder using the sampling mode [`Full`](@ref).
# If you want to compute the "disorder" for a specific threshold, it is possible using
# the internal struct `PartialDisorder`:

partial = RecurrenceMicrostatesAnalysis.PartialDisorder(ThresholdRecurrence(ε), N)
complexity(partial, X)

# ## Custom specification of recurrence microstates

# When we write `rmspace = RecurrenceMicrostates(ε, N)`,
# we are in fact accepting a default definition for both what counts as a recurrence
# and which recurrence microstates to examine.
# We can alter either by choosing the recurrence expression, the specific
# microstate(s) we wish to analyze, or the sampling method used to extract these
# microstates from the input data. For example:

expr = CorridorRecurrence(0.05, 0.27)
shape = TriangleMicrostate(3)
sampling = Full()
rmspace = RecurrenceMicrostates(expr, shape; sampling)
probabilities(rmspace, X)

# **RecurrenceMicrostateAnalysis.jl** supports several configurations for the recurrence outcome space
# while leveraging the same backend (see [`RecurrenceMicrostates`](@ref)).
# If you want to contribute with new recurrence expressions, microstate shapes, or sampling modes,
# read the section [for devs](@ref devs) and open an 
# [issue](https://github.com/JuliaDynamics/RecurrenceMicrostatesAnalysis.jl/issues)
# if you encounter any difficulties 🙂

# ## Cross recurrence plots

# For cross-recurrences, nearly nothing changes for you, nor for the source code
# of the code base! Simply call `function(..., rmspace, X, Y)`, adding an additional
# final argument `Y` corresponding to the second trajectory from which cross recurrences are estimated.

# For example, here are the cross recurrence microstate distribution for
# the original Henon map trajectory and one at slightly different parameters

set_parameter!(henon, 1, 1.35)
Y, t = trajectory(henon, 10_000)
probabilities(rmspace, X, Y)

# This augmentation from one to two input data
# works for most functions discussed in this tutorial.
# Coincidentally, the same extension of `probabilities` to multivariate data
# is done in [Associations.jl](https://juliadynamics.github.io/Associations.jl/stable/).

# ## Spatial data

# Finally, let's discuss spatial data. This is an exploratory
# method implemented in the package based on Spatial Recurrence Plots [Marwan2007Spatial](@cite).
# It means that the microstates can be a tensor structure, e.g., a hypercube.
# However, it is also important to note that the number of bits (or recurrences)
# inside the microstate increases, resulting in an exponential increase in
# the probability distribution length, which can result in a lack of memory.

# To exemplify its use, we will define here 2D square microstates $3\times 3$,
# but that is a projection of the tensorial hypercubic microstate with side length $3$
# into the first and third dimensions; that is:

shape = (3, 1, 3, 1)
srmspace = RecurrenceMicrostates(ε, shape)

# As an example, let's use an RGB image:

import Images
img = Images.load("assets/example.jpg")

# We need to reorganize it as an `Array{3, Float64}`. The first dimension
# must be our RGB values, while the other two need to be the horizontal and
# vertical positions of the pixels.

W, H = size(img)
arr = Array{Float64}(undef, 3, H, W)
for i in 1:H, j in 1:W
    c = img[i, j]
    arr[1, i, j] = float(c.r)
    arr[2, i, j] = float(c.g)
    arr[3, i, j] = float(c.b)
end

size(arr) == (3, H, W)

# Finally, we can use the `probabilities` function to estimate our recurrence
# microstate distribution.

probs = probabilities(srmspace, arr)

# Although Marwan adapted some RQA quantities for spatial recurrence plots,
# they cannot be estimated using RMA. The only exception here is the recurrence
# rate, which can be estimated as:

RecurrenceMicrostatesAnalysis.measure(rr_estimator, probs)

# Another quantity that can be computed is the recurrence microstate entropy:

entropy(Shannon(), probs)

# RMA with spatial data is a very interesting and complex topic, and we
# have implemented it to motivate possible research using this feature.
# So feel free to try it and notify us if you have some success 😁

# Note that if you are using a grayscale image, you need to use an
# `Array` with size `(1, H, W)`. The first dimension stores the
# features of the data, which are used to compute the recurrences,
# i.e., $\vec{x}_{\vec{i}}$. The same principle must be applied 
# to other types of spatial data.

# ##   GPU

# Computation of recurrence microstate distributions via **RecurrenceMicrostatesAnalysis.jl**
# is compatible with different GPU devices due to an abstract kernel written using
# **KernelAbstractions.jl**.

# The use of the GPU backend is very similar to the CPU backend; however, the input type
# must be an `AbstractGPUVector{<:SVector}`, instead of a [`StateSpaceSet`](@ref) or a `Vector{<:Real}`.

# !!! compat "Spatial data"
#     The current GPU backend is not compatible with spatial data. It only works with
#     time series.

# ### Computing recurrence microstate distributions

# The first step to compute a recurrence microstate distribution using a GPU is to
# import the package associated with your device, such as **CUDA.jl** or **Metal.jl**.
# Next, you need to move your [`StateSpaceSet`](@ref) to the GPU. For example:

# ```julia
# using CUDA
# 
# X_gpu = Float32.(X[:, 1]) |> StateSpaceSet |> CuVector
# ```

# !!! compat "Float type"
#     GPUs usually only accept `Float32`.

# This GPU vector `X_gpu` can be used as input for the `probabilities` function:

# ```julia
# ε = 0.27f0
# N = 3
# rmspace = RecurrenceMicrostates(ε, N; metric = GPUEuclidean())
# probs = probabilities(rmspace, X_gpu)
# ```

# ```julia
# Probabilities{Float64,1} over 512 outcomes
#    1  0.1298675475359977
#    2  0.017550106511280954
#    3  0.028929580130109996
#    4  0.0016157228214197196
#    5  0.039249842118455266
#    6  0.014588514785254093
#    ⋮  
#  507  0.0001702340127557486
#  508  0.0006545307752488948
#  509  0.00010442086328848504
#  510  0.00043848760982444294
#  511  0.0
#  512  0.0031636320936923195
# ```

# Note that the recurrence microstate outcome space has two specifications:

# 1. The `threshold` must have the same type as the input. If you have an `AbstractGPUVector{<:SVector{D, T}}` as input,
#    where `T` is the data type (e.g., `Float32` or `Float64`), your `threshold` must also be of type `T`.

# 2. The `metric` must be a [`GPUMetric`](@ref). This is required because **Distances.jl** is not fully compatible with GPUs.

# !!! compat "Sampling ratio"
#     When using a random sampling mode (e.g., [`SRandom`](@ref)), the samples are extracted
#     on the CPU, and only the microstates are computed on the GPU. Therefore, the sampling ratio
#     can be `Float64`, even if the GPU is not compatible with this data type.

# ### Estimating RQA and disorder

# Just as the distribution computation keeps the same structure when using a GPU instead of a CPU,
# the estimation of RQA or the computation of disorder or recurrence entropy is similar.

# - Entropy
# ```julia
# entropy(Shannon(), rmspace, X_gpu)
# ```

# - Recurrence rate
# ```julia
# complexity(RecurrenceRate(ε; metric = GPUEuclidean()), X_gpu)
# ```

# - Determinism
# ```julia
# complexity(RecurrenceDeterminism(ε; metric = GPUEuclidean()), X_gpu)
# ```

# - Laminarity
# ```julia
# complexity(RecurrenceLaminarity(ε; metric = GPUEuclidean()), X_gpu)
# ```

# - Disorder
# ```julia
# complexity(Disorder(N; metric = GPUEuclidean()), X_gpu)
# ```

# - Windowed disorder
# ```julia
# W = 100
# complexity(WindowedDisorder(W, N; metric = GPUEuclidean()), X_gpu)
# ```

# !!! warning "Time-series length"
#     Note that if you are using `WindowedDisorder` for a long time series, but splitting
#     it into small windows (e.g, `W = 1000`), the GPU can be less effienct than the CPU.
#     It happens because only the probability computation is performed in the GPU.

# ### Performance

# Working with RMA on a GPU is faster than using a CPU. However, it is
# important to note that GPUs require initialization time; therefore, they perform
# better for long time series 🙂

# Let's test it using **BenchmarkTools.jl**!
using BenchmarkTools

# !!! info "Hardware"
#     The tests performed in this section was done using an Apple MacBook M1
#     with 8GM RAM.

# First, we can compute some probability distributions on the CPU. To make it
# expensive, we use microstates $5\times 5$ and take all microstates
# available in the RP. Then, for a time series of length 1,000 points,
# we have ~1,000,000 microstates, and since these microstates have 25
# recurrences, we need to compute approximately 25 million recurrences.
# For a time series with 5,000 points we have ~5 times this value 😉
# ```julia
# X_test = X[1:1000, 1] |> StateSpaceSet
# rmspace = RecurrenceMicrostates(0.27, 5; sampling = Full())
# @benchmark probabilities(rmspace, X_test)
# ```
# ```julia
# BenchmarkTools.Trial: 4 samples with 1 evaluation per sample.
#  Range (min … max):  1.359 s …    1.671 s  ┊ GC (min … max): 17.24% … 30.81%
#  Time  (median):     1.585 s               ┊ GC (median):    27.97%
#  Time  (mean ± σ):   1.550 s ± 147.209 ms  ┊ GC (mean ± σ):  26.38% ±  6.30%

#   █                          █                           █ █  
#   █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁█ ▁
#   1.36 s         Histogram: frequency by time         1.67 s <

#  Memory estimate: 4.00 GiB, allocs estimate: 109.
# ```

# ```julia
# X_test = X[1:5000, 1] |> StateSpaceSet
# @benchmark probabilities(rmspace, X_test)
# ```
# ```julia
# BenchmarkTools.Trial: 3 samples with 1 evaluation per sample.
#  Range (min … max):  1.757 s …   1.909 s  ┊ GC (min … max): 23.06% … 28.67%
#  Time  (median):     1.804 s              ┊ GC (median):    23.96%
#  Time  (mean ± σ):   1.823 s ± 77.884 ms  ┊ GC (mean ± σ):  25.31% ±  3.01%

#   █                █                                      █  
#   █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁█ ▁
#   1.76 s         Histogram: frequency by time        1.91 s <

#  Memory estimate: 4.00 GiB, allocs estimate: 109.
# ```

# Now, let's perform the same example on the GPU:
# ```julia
# using Metal
# X_gpu = Float32.(X[1:1000, 1]) |> StateSpaceSet |> MtlVector
# rmspace = RecurrenceMicrostates(0.27f0, 5; sampling = Full(), metric = GPUEuclidean())
# @benchmark probabilities(rmspace, X_gpu)
# ```
# ```julia
# BenchmarkTools.Trial: 41 samples with 1 evaluation per sample.
#  Range (min … max):   52.483 ms … 212.468 ms  ┊ GC (min … max):  0.00% … 57.84%
#  Time  (median):     120.866 ms               ┊ GC (median):     3.71%
#  Time  (mean ± σ):   121.944 ms ±  43.234 ms  ┊ GC (mean ± σ):  24.97% ± 24.62%

#   ▃         ▃█            ▃ ▃    ▃▃█  ▃▃                      ▃  
#   █▇▇▇▁▁▁▁▁▇██▁▇▇▁▁▁▇▇▁▇▁▁█▇█▇▁▁▇███▁▁██▁▇▁▁▇▁▁▁▁▁▁▇▁▁▁▁▁▁▇▇▁▁█ ▁
#   52.5 ms          Histogram: frequency by time          212 ms <

#  Memory estimate: 640.02 MiB, allocs estimate: 474.
# ```

# ```julia
# X_gpu = Float32.(X[1:5000, 1]) |> StateSpaceSet |> MtlVector
# @benchmark probabilities(rmspace, X_gpu)
# ```
# ```julia
# BenchmarkTools.Trial: 25 samples with 1 evaluation per sample.
#  Range (min … max):  135.172 ms … 260.022 ms  ┊ GC (min … max):  0.95% … 44.50%
#  Time  (median):     204.136 ms               ┊ GC (median):    17.25%
#  Time  (mean ± σ):   204.717 ms ±  29.069 ms  ┊ GC (mean ± σ):  22.71% ± 20.08%

#   ▁     ▁          ▁  █▁       █▁▁█▁▁  ▁▁ ▁  █▁▁▁▁         ▁  ▁  
#   █▁▁▁▁▁█▁▁▁▁▁▁▁▁▁▁█▁▁██▁▁▁▁▁▁▁██████▁▁██▁█▁▁█████▁▁▁▁▁▁▁▁▁█▁▁█ ▁
#   135 ms           Histogram: frequency by time          260 ms <

#  Memory estimate: 640.02 MiB, allocs estimate: 474.
# ```

# The results show a considerable difference in computational time
# between the two cases. The CPU requires around 10 times more time
# to compute the same task as the GPU 🙂

# Let's also run a test using the disorder computation:
# ```julia
# X_test = X[1:1000, 1] |> StateSpaceSet
# @benchmark complexity(Disorder(4), X_test)
# ```
# ```julia
# BenchmarkTools.Trial: 20 samples with 1 evaluation per sample.
#  Range (min … max):  203.483 ms … 327.038 ms  ┊ GC (min … max):  2.04% … 26.77%
#  Time  (median):     291.676 ms               ┊ GC (median):    29.13%
#  Time  (mean ± σ):   264.279 ms ±  43.102 ms  ┊ GC (mean ± σ):  19.37% ± 13.55%

#   ▃                                            █▃                
#   █▁▁▇▁▁▇▇▇▇▁▇▁▁▁▁▁▁▁▁▁▁▁▇▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▇▇██▁▁▇▇▇▁▁▁▁▁▁▁▁▇ ▁
#   203 ms           Histogram: frequency by time          327 ms <

#  Memory estimate: 541.90 MiB, allocs estimate: 600351.
# ```

# ```julia
# X_gpu = Float32.(X[1:1000, 1]) |> StateSpaceSet |> MtlVector
# @benchmark complexity(Disorder(4; metric = GPUEuclidean()), X_gpu)
# ```
# ```julia
# BenchmarkTools.Trial: 16 samples with 1 evaluation per sample.
#  Range (min … max):  240.116 ms … 390.540 ms  ┊ GC (min … max): 0.68% … 21.21%
#  Time  (median):     324.074 ms               ┊ GC (median):    2.73%
#  Time  (mean ± σ):   324.189 ms ±  44.783 ms  ┊ GC (mean ± σ):  9.93% ±  9.99%

#   ▁        ▁     ▁ ▁        ▁  ▁▁  ▁█▁             ▁  ▁ ▁     █  
#   █▁▁▁▁▁▁▁▁█▁▁▁▁▁█▁█▁▁▁▁▁▁▁▁█▁▁██▁▁███▁▁▁▁▁▁▁▁▁▁▁▁▁█▁▁█▁█▁▁▁▁▁█ ▁
#   240 ms           Histogram: frequency by time          391 ms <

#  Memory estimate: 268.43 MiB, allocs estimate: 614832.
# ```

# Note that in this situation, the CPU is slightly faster than the GPU due to
# several internal initializations that are not offset by the computation time.

# Now, let's try the same test using a larger time series:
# ```julia
# X_test = X[1:9000, 1] |> StateSpaceSet
# @benchmark complexity(Disorder(4), X_test)
# ```
# ```julia
# BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
#  Single result which took 13.837 s (1.41% GC) to evaluate,
#  with a memory estimate of 534.26 MiB, over 600348 allocations.
# ```

# ```julia
# X_gpu = Float32.(X[1:9000, 1]) |> StateSpaceSet |> MtlVector
# @benchmark complexity(Disorder(4; metric = GPUEuclidean()), X_gpu)
# ```
# ```julia
# BenchmarkTools.Trial: 1 sample with 1 evaluation per sample.
#  Single result which took 7.427 s (0.14% GC) to evaluate,
#  with a memory estimate of 264.65 MiB, over 614902 allocations.
# ```

# And now, computation is faster on the GPU than on the CPU. Therefore, it is important
# to note that the GPU is not a magic solution: for long time series when computing
# recurrence microstate probabilities, the GPU will be faster; however, for smaller cases,
# it may be better to use the CPU.