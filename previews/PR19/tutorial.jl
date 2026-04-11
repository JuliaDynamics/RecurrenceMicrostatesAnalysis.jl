# # Tutorial for RecurrenceMicrostatesAnalysis.jl

# In this tutorial we go through a typical usage of  **RecurrenceMicrostatesAnalysis.jl**.
# We'll see how to calculate distributions of recurrence microstates,
# how to optimize our choices regarding the distribution generation,
# and how to perform Recurrence Microstate Analysis (**RMA**).

# !!! info "ComplexityMeasures.jl"
#     RecurrenceMicrostatesAnalysis.jl interfaces with, and extends, ComplexityMeasures.jl.
#     It can enhance your understanding if you have first view the [tutorial of ComplexityMeasures.jl](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/tutorial/). 
#     Regardless the current tutorial is written to be self-contained.

# ## Crash-course into RMA

# Recurrence Plots (RPs) were introduced in 1987 by Eckmann et al.
# [Eckmann1987RP](@cite) as a method for analyzing dynamical systems through recurrence
# properties.

# Consider a time series $\vec{x}_i \in \mathbb{R}^d$, $i \in \{1, 2, \dots, K\}$,
# where $K$ is the length of the time series and $d$ is the dimension of the phase space.
# The recurrence plot is defined by the recurrence matrix
# ```math
# r_{(i,j)} = \Theta(\varepsilon - \|\vec x_i - \vec x_j\|),
# ```
# where $\Theta(\cdot)$ denotes the Heaviside step function and $\varepsilon$ is the recurrence
# threshold.

# The following figure shows examples of recurrence plots for different systems:
# (a) white noise;
# (b) a superposition of harmonic oscillators;
# (c) a logistic map with a linear trend;
# (d) Brownian motion.

# ![Image of four RPs with their timeseries](assets/rps.png)

# A recurrence microstate is a local structure extracted from an RP. For a given microstate
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
# Other types are also possible as we described in [Input data for RecurrenceMicrostatesAnalysis.jl](@ref).

# Now, we specify the recurrence microstate configuration

using RecurrenceMicrostatesAnalysis, Distances

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
# the number of thresholds used, or its range). This is because the quantifier is
# defined as the maximum total entropy of recurrence microstate classes, considering
# a large range of thresholds. Moreover, this quantifier can only be estimated
# through recurrence microstates.

# It is also possible to estimate disorder while splitting the data into windows.
# We prepared a special complexity estimator for this, aiming to facilitate
# its usage. In this situation, you must define a [`WindowedDisorder`](@ref).

window_len = 1000
win_disorder = WindowedDisorder(window_len; step = 100)

# Here, we are using windows of length 1000 points, moved in steps of 100 points.
# Finally, we compute the quantifier:

wd = complexity(win_disorder, X)

# Plotting it:
using CairoMakie
lines(wd)

# ## Optimization of free parameters

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
# i.e., $\vec{x}_{\vec{i}}$.