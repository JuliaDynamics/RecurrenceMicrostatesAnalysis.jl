# # Examples

# In this section, we provide some examples where it is possible
# to apply RMA to analyze data.

# ## Classifying data with a multi-layer perceptron and RMA

# In this example, we demonstrate how to use a multi-layer perceptron with
# RMA to classify time series based on a parameter used to generate them.
# It is based on [Spezzatto2024ML](@cite), and we will use the
# package **Flux.jl** to perform the machine learning tasks.

# ### Generating data

# Let's begin by generating a dataset. Here, we will use the Lorenz system.
# First, we can prepare some utilities to help generate our data:

using DynamicalSystemsBase, PredefinedDynamicalSystems

ρ_cls = [26.0, 27.0, 28.0, 29.0, 30.0]
num_samples_to_test = 50
num_samples_to_train = 200

train_timeseries = Vector{StateSpaceSet}(undef, length(ρ_cls) * num_samples_to_train)
test_timeseries = Vector{StateSpaceSet}(undef, length(ρ_cls) * num_samples_to_test)

train_labels = Vector{Float64}(undef, length(ρ_cls) * num_samples_to_train)
test_labels = Vector{Float64}(undef, length(ρ_cls) * num_samples_to_test)

function data_traj(ρ; u0 = rand(3), t = 250.0, Ttr = 1200.0, Δt = 0.2)
    lorenz = PredefinedDynamicalSystems.lorenz(u0; ρ)
    x, _ = trajectory(lorenz, t; Ttr, Δt)
    return x
end

function generate_data!(labels, data, classes, num_elements_per_class)
    c = 1
    for i ∈ eachindex(labels)
        labels[i] = classes[c]
        data[i] = data_traj(classes[c])

        if (i % num_elements_per_class == 0)
            c += 1
        end
    end
end

# Then, we can generate a training dataset:

generate_data!(train_labels, train_timeseries, ρ_cls, num_samples_to_train)
train_timeseries

# And a test dataset:

generate_data!(test_labels, test_timeseries, ρ_cls, num_samples_to_test)
test_timeseries

# ### Preparing the input features

# Our feature is the recurrence microstate distribution. Therefore, for each time series
# we must compute the probabilities and store them as a feature vector:

using RecurrenceMicrostatesAnalysis

N = 3
train_dataset = Matrix{Float64}(undef, 2^(N * N) + 2, length(train_labels))
test_dataset = Matrix{Float64}(undef, 2^(N * N) + 2, length(test_labels))

function get_probs!(dataset, timeseries, N)
    for i ∈ eachindex(timeseries)
        th, s = optimize(Threshold(), Shannon(), N, timeseries[i])
        rmspace = RecurrenceMicrostates(th, N)
        dist = probabilities(rmspace, timeseries[i])
        dataset[1, i] = th
        dataset[2, i] = s
        dataset[3:end, i] = dist[1:end]
    end
end

# Note that we are also using the recurrence threshold and the recurrence entropy as features.
# This follows [Spezzatto2024ML](@cite), but it is not strictly necessary. If your input
# has a stable threshold range that maximizes the recurrence entropy, it is possible to use
# a mean threshold to compute all probability distributions and use only these distributions
# as features for the machine learning model.

# Computing our feature vectors:

# - Train

get_probs!(train_dataset, train_timeseries, N)
train_dataset

# - Test

get_probs!(test_dataset, test_timeseries, N)
test_dataset

# ### Defining the neural network model

# Here, we use **Flux.jl** to define our model. As mentioned before, we are using a
# multi-layer perceptron. However, it is also possible to use other approaches, e.g.,
# random forests [Silveira2025ML](@cite).

using Flux

model = Chain(
    Dense(2^(N * N) + 2 => 512, identity),
    Dense(512 => 256, selu),
    Dense(256 => 64, selu),
    Dense(64 => length(ρ_cls)),
    softmax
)

model = f64(model)

# ### Training the MLP

# Now it is just standard machine learning procedure...

train_labels = Flux.onehotbatch(train_labels, ρ_cls)
test_labels = Flux.onehotbatch(test_labels, ρ_cls)

loader = Flux.DataLoader((train_dataset, train_labels), batchsize = 32, shuffle = true)
opt = Flux.setup(Flux.Adam(0.005), model)

for epc ∈ 1:50
    for (x, y) ∈ loader
        _, grads = Flux.withgradient(model) do m
            y_hat = m(x)
            Flux.crossentropy(y_hat, y)
        end

        Flux.update!(opt, model, grads[1])
    end
end

# ### Model evaluation

# Finally, let's check our accuracy 🙂

using LinearAlgebra

function get_quantifiers(predict, trusty, classes)
    conf = zeros(Int, length(classes), length(classes))
    sz = size(predict, 2)

    for i in 1:sz
        mx_prd = findmax(predict[:, i])
        mx_trt = findmax(trusty[:, i])

        conf[mx_prd[2], mx_trt[2]] += 1
    end

    return tr(conf) / (sum(conf) + eps())
end

accuracy = get_quantifiers(model(test_dataset), test_labels, ρ_cls)
accuracy * 100