#   GPU

Computation of recurrence microstate distributions via **RecurrenceMicrostatesAnalysis.jl**
is compatible with different GPU devices due to an abstract kernel written using
**KernelAbstractions.jl**.

The use of the GPU backend is very similar to the CPU backend; however, the input type
must be an `AbstractGPUVector{<:SVector}`, instead of a [`StateSpaceSet`](@ref) or a `Vector{<:Real}`.

!!! compat "Spatial data"
    The current GPU backend is not compatible with spatial data. It only works with
    time series.

## Computing recurrence microstate distributions

The first step to compute a recurrence microstate distribution using a GPU is to
import the package associated with your device, such as **CUDA.jl** or **Metal.jl**.
Next, you need to move your [`StateSpaceSet`](@ref) to the GPU. For example:

```julia
using CUDA, DynamicalSystemsBase, PredefinedDynamicalSystems

logistic = PredefinedDynamicalSystems.logistic(0.4; r = 4.0)
X, t = trajectory(logistic, 1000; Ttr = 2000)

X_gpu = Float32.(X[:, 1]) |> StateSpaceSet |> CuVector
```

!!! compat "Float type"
    GPUs usually only accept `Float32`.

This GPU vector `X_gpu` can be used as input for the `probabilities` function:

```julia
using RecurrenceMicrostatesAnalysis

ε = 0.27f0
N = 3
rmspace = RecurrenceMicrostates(ε, N; metric = GPUEuclidean())
probs = probabilities(rmspace, X_gpu)
```

Note that the recurrence microstate outcome space has two specifications:

1. The `threshold` must have the same type as the input. If you have an `AbstractGPUVector{<:SVector{D, T}}` as input,
   where `T` is the data type (e.g., `Float32` or `Float64`), your `threshold` must also be of type `T`.

2. The `metric` must be a [`GPUMetric`](@ref). This is required because **Distances.jl** is not fully compatible with GPUs.

!!! compat "Sampling ratio"
    When using a random sampling mode (e.g., [`SRandom`](@ref)), the samples are extracted
    on the CPU, and only the microstates are computed on the GPU. Therefore, the sampling ratio
    can be `Float64`, even if the GPU is not compatible with this data type.

## Estimating RQA and disorder

Just as the distribution computation keeps the same structure when using a GPU instead of a CPU,
the estimation of RQA or the computation of disorder or recurrence entropy is similar.

- Entropy
```julia
entropy(Shannon(), rmspace, X_gpu)
```

- Recurrence rate
```julia
complexity(RecurrenceRate(ε; metric = GPUEuclidean()), X_gpu)
```

- Determinism
```julia
complexity(RecurrenceDeterminism(ε; metric = GPUEuclidean()), X_gpu)
```

- Laminarity
```julia
complexity(RecurrenceLaminarity(ε; metric = GPUEuclidean()), X_gpu)
```

- Disorder
```julia
complexity(Disorder(N; metric = GPUEuclidean()), X_gpu)
```

- Windowed disorder
```julia
W = 100
complexity(WindowedDisorder(W, N; metric = GPUEuclidean()), X_gpu)
```

!!! info "Performance"
    Working with RMA on a GPU is faster than using a CPU, as expected. However, it is
    important to note that GPUs require initialization time; therefore, they perform
    better for long time series 🙂