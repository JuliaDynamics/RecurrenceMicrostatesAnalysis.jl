# # Tutorial for RecurrenceMicrostatesAnalysis.jl

# In this tutorial we go through a typical usage of  **RecurrenceMicrostatesAnalysis.jl**.
# We'll see how to calculate distributions of recurrence microstates,
# how to optimize our choices regarding the distribution generation,
# and how to perform Recurrence Microstate Analysis (RMA).

# !!! info "ComplexityMeasures.jl"
#     RecurrenceMicrostatesAnalysis.jl interfaces with, and extends, ComplexityMeasures.jl.
#     It can enhance your understanding if you have first view the tutorial of
#     ComplexityMeasures.jl. Regardless the current tutorial is written to be self-contained.

# ## Crash-course into RMA

# Recurrence Plots (RPs) were introduced in 1987 by Eckmann et al.
#  [Eckmann1987RP](@cite) as a method for analyzing dynamical systems through recurrence
# properties.

# Consider a time series $\vec{x}_i \in \mathbb{R}^d$, $i \in \{1, 2, \dots, K\}$,
# where $K$ is the length of the time series and $d$ is the dimension of the phase space.
# The recurrence plot is defined by the recurrence matrix
# ```math
# R_{i,j} = \Theta(\varepsilon - \|\vec x_i - \vec x_j\|),
# ```
# where $\Theta(\cdot)$ denotes the Heaviside step function and $\varepsilon$ is the recurrence
# threshold.

# The following figure shows examples of recurrence plots for different systems:
# (a) white noise;
# (b) a superposition of harmonic oscillators;
# (c) a logistic map with a linear trend;
# (d) Brownian motion.

# ![Image of four RPs with their timeseries](../assets/rps.png)

# A recurrence microstate is a local structure extracted from an RP. For a given microstate
# shape and size, the set of possible microstates is finite. For example, square microstates
# with size $N = 2$ yield $16$ distinct configurations.

# ![Image of the 16 squared microstates to N = 2](../assets/microstates.png)

# Recurrence Microstates Analysis (RMA) uses the probability distribution of these microstates
# as a source of information for characterizing dynamical systems.

# ## Probability distributions of recurrence microstates

# Finding and counting microstates in data is straightforward.
# It amounts to passing the input data to the `probabilities` function,
# while specifying the options of the  `RecurrenceMicrostates` estimator,
# which essentially means e.g., what sort of distance threshold defines a recurrence,
# and what is maximum microstate size to consider.

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

total_time = 10_000
X, t = trajectory(henon, total_time)
X

# Notice that `X` is already a [`StateSpaceSet`](@ref). Because  **RecurrenceMicrostatesAnalysis.jl**
# is part of **DynamicalSystems.jl**, this data type is the preferred input type.
# Other types are also possible however, see the documentation of the
# [`RecurrenceMicrostates`](@ref) central type for more.

# Now, we specify the recurrence microstate configuration

ε = 0.27
N = 2
rmspace = RecurrenceMicrostates(ε, N)

# and finally call

probs = probabilities(ospace, X)

# The [`probability`](@ref) function is the same function as in [`ComplexityMeasures`](@ref).
# Given an outcome space, that is a way to _symbolize_ input data into discrete outcomes,
# `probabilities` return the probability (relative occurrence frequency) for each outcome.
# And indeed, the recurrence microstates is an outcome space.

# If instead of the probabilities of the microstates you want their direct count
# simply replace `probabilities` with `counts`.

counts(rmspace, X)


# ## Recurrence microstates analysis (RMA)

# To actually analyze your data, there are two ways forwards.
# One way, is to utilize these probabilities within the interface provided
# by [`ComplexityMeasures`](@ref) to calculate entropies.
# For example, the corresponding Shannon entropy is

entropy(Shannon(), probs)

# (note that the API of `ComplexityMeasures` is re-exported by `RecurrenceMicrostateAnalysis`).
# This number corresponds to the **recurrence microstate entropy** as defined in our
# publication [`WhichPaperIscorrectToCite`](@cite).

# `ComplexityMeasures` allows the convenience syntax of

entropy(Shannon(), rmspace, X)

# so in this case we wouldn't need to calculate the probabilities directly.
# Naturally, any other entropy could be estimated instead,

entropy(Tsallis(), rmspace, X)

# although we haven't explored alternative entropies in research yet.

# The secon way forwards is the more traditional recurrence quantification analysis
# route, where you estimate (approximate really) various quantities
# such as laminarity that fundamentally relate with the context of recurrences.
# For example,

# These quantities are listed in XXX.

# Note that if instead of



# ## Optimizing recurrence specification

# In the above example we blindly selected the recurrence threshold `ε`.
# A better approach is to optimize it, so it (for example) maximizes
# the recurrence microstate entropy.
# This can be done with the [`optimize`](@ref) function

ε, S = optimize(Threshold(), RecurrenceEntropy(), X, N)
rmspace = RecurrenceMicrostates(ε, N)
h = entropy(Shannon(), rmspace, X)
(h, S)


# ## Custom specification of recurrence microstates

# When we write `rmspace = RecurrenceMicrostates(ε, N)`,
# we are in fact accepting a default definition for both what counts as a recurrence
# as well as what recurrence microstates to examine.
# We can alter either, by choosing the recurrence expression, or the specific
# microstate(s) we wish to analyze. For example

expr = CorridorRecurrence(0.05, 0.27)
shape = MicrostateTriangle(lalala)
rmspace = RecurrenceMicrostates(; expression = expr, shape)
probabilities(rmspace, X)

# More details are given in [`RecurrenceMicrostates`](@ref)
# and the [API](@ref) section of the docs.


# ## Cross recurrence plots

# For cross-recurrences, nearly nothing changes for you, nor for the source code
# of the code base! Simply call `function(..., rmspace, X, Y)`, adding an additional
# final argument `Y` corresponding to the second trajectory from which cross recurrences are estimated.

# For example, here are the cross recurrence microstate distribution for
# the original Henon map trajectory and one at slightly different parameters

set_parameter!(henon, 1, 1.35)
Y, t = trajectory(henon, total_time)
probabilities(rmspace, X, Y)

# This augmentation from one to two input data
# works for all functions discussed in this tutorial.

# ## Spatial data
