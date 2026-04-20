#   [RecurrenceMicrostatesAnalysis.jl for devs](@id devs)

!!! tip
    All pull requests that introduce new functionality must be thoroughly tested and documented.
    Tests are required only for the methods you extend. Always remember to add docstrings to
    your implementations, as well as tests to validate them.

    We recommend reading the
    [Good Scientific Code Workshop](https://github.com/JuliaDynamics/GoodScientificCodeWorkshop)

## Backend of RecurrenceMicrostatesAnalysis.jl

**RecurrenceMicrostatesAnalysis.jl** has multiple backends, depending on the usage context.
Each backend is implemented based on an internal struct `RMACore` or on the input data type.

There are three backend implementations:

- For sequential data using the CPU: it internally uses a `CPUCore <: RMACore`, which defines that
    the process must run on the CPU, and the input data is a [`StateSpaceSet`](@ref) (or a `Vector{<:Real}`).

- For spatial data using the CPU: it internally uses a `CPUCore <: RMACore`, which defines that
    the process must run on the CPU, and the input data is an `Array`.

- For sequential data using the GPU: it internally uses a `GPUCore <: RMACore`, which defines that
    the process must run on the GPU, and the input data is an `AbstractGPUVector{<:SVector}`.

Note that there is a significant difference between the input data types, in such a way that
`RMACore` is just an auxiliary struct used to differentiate the hardware backend internally.

!!! info "Backend"
    The package backends are located at `src/core/cpu_core.jl` (CPU) and `src/core/gpu/gpu_core.jl` (GPU).

You do not need to fully understand how the backend operates to define something new in the package.
However, it is important to understand how the backend requires internal functions based on
`RMACore` or the input data type. For example, when implementing a [`RecurrenceExpression`](@ref),
the CPU structure uses a [`recurrence`](@ref) function, while the GPU structure uses a
`gpu_recurrence` function.

## Adding a new Recurrence Expression

### Steps

1. Define the mathematical expression of your recurrence expression. It must return a binary value:
    `UInt(0)` for non-recurrence and `UInt(1)` for recurrence.
2. Define a new type `YourType <: RecurrenceExpression`. Constant parameters (e.g., recurrence threshold and metric)
    should be fields of this type.
3. Implement the appropriate `recurrence` dispatch:
    - Sequential data: `recurrence(expr::YourType, x::StateSpaceSet, y::StateSpaceSet, i::Int, j::Int)`
    - Spatial data: `recurrence(expr::YourType, x::AbstractArray{<:Real}, y::AbstractArray{<:Real}, i::NTuple{N,Int}, j::NTuple{M,Int})`
    - GPU: `gpu_recurrence(expr::YourType, x, y, i, j, n)`
4. Add a docstring describing the mathematical definition and relevant references.
5. Add the recurrence expression to `docs/src/api.md`.
6. Add the expression to the [`RecurrenceExpression`](@ref) docstring.
7. Add tests to `test/distributions.jl` and `test/recurrences.jl`.

### Example

Let's define a "recurrence" expression as:
```math
r_{(i,j)} = \Theta(\|\vec{x}_i - \vec{x}_j\| - \varepsilon).
```

First, we define our struct:
```@example dev
using RecurrenceMicrostatesAnalysis
using Distances: Euclidean, Metric, evaluate

struct MyRecurrenceExpr{T <: Real, M <: Metric} <: RecurrenceExpression{T, M}
    ε::T
    metric::M
end

MyRecurrenceExpr(ε) = MyRecurrenceExpr(ε, Euclidean())
```

Next, we define the recurrence function:
```@example dev
@inline function RecurrenceMicrostatesAnalysis.recurrence(
    expr::MyRecurrenceExpr,
    x::StateSpaceSet,
    y::StateSpaceSet,
    i::Int,
    j::Int,
)
    distance = @inbounds evaluate(expr.metric, x.data[i], y.data[j])
    return UInt8(distance ≥ expr.ε)
end
```

And that's it:
```@example dev
rmspace = RecurrenceMicrostates(MyRecurrenceExpr(0.27), 3)
```

```@example dev
X = randn(1000) |> StateSpaceSet
probabilities(rmspace, X)
```

## Adding a new Sampling Mode

### Steps

1. Define how the sampling mode operates: which microstates are sampled, from which regions, and in what quantity.
    The [`SamplingSpace`](@ref) must be taken into account when designing the sampling logic.
2. Define a new struct that is a subtype of [`SamplingMode`](@ref). The struct may be empty (e.g., [`Full`](@ref)) or
    contain parameters such as a sampling ratio (e.g., [`SRandom`](@ref)).
3. Implement the dispatch `get_num_samples(mode::YourType, ::SamplingSpace)`, which determines the number of samples
    to be drawn given the sampling mode and the sampling space.
4. Implement the dispatch `get_sample(::RMACore, ::YourType, space::SamplingSpace, rng, m)`, which returns the starting
    pair $(i, j)$ to construct the microstate. Here, `RMACore` defines whether it is running on the CPU or GPU, `rng` is
    a random number generator, and `m` is a counter of microstates.
5. Add a docstring to your sampling mode describing its behavior and initialization.
6. Add your sampling mode to `docs/src/api.md`.
7. Add the expression to the [`SamplingMode`](@ref) docstring.
8. Add tests to `test/distributions.jl` and `test/sampling.jl`.

## Adding a new Microstate Shape

### Steps

1. Define your microstate design. It essentially determines the microstate structure and reading order. For example,
    square microstates are read row-wise, while triangular microstates may be read column-wise. Each position in
    the microstate structure must be associated with a power of two in order to convert the binary microstate into
    a decimal index.
2. Define a new struct that is a subtype of [`MicrostateShape`](@ref).
3. Implement the dispatch `get_histogram_size(::MyShape)`, which returns the histogram length.
4. Implement the dispatch `get_power_vector(::RMACore, ::MyShape)`, which returns the power vector used to
    read the microstate as an integer.
5. Implement the dispatch `get_offsets(::RMACore, ::MyShape)`, which returns which positions are accessed from $(i, j)$
    to construct the microstate.
6. Define how your shape reacts to a [`SamplingSpace`](@ref) by implementing `SamplingSpace(::MyType, x, y)`.
7. Add a docstring to your microstate shape describing its behavior and initialization.
8. Add your microstate shape to `docs/src/api.md`.
9. Add the expression to the [`MicrostateShape`](@ref) docstring.
10. Add tests to `test/distributions.jl` and `test/shapes.jl`.

### Example

As an example, let's try to construct the struct to obtain the microstate:
```math
\begin{matrix}
r_{(i,j)} & & r_{(i, j+2)} \\
 & r_{(i+1, j+1)} & \\ 
 r_{(i+2,j)} & & r_{(i+2, j+2)}
\end{matrix}
```

For simplicity, we are not going to generalize it for arbitrary sizes 😅. First, let's define our struct:
```@example dev
struct MyMicrostateShape <: MicrostateShape end
```

Next, we need to define the histogram size, which will be returned by `get_histogram_size`. It is $2^\sigma$,
where $\sigma$ is the number of recurrences contained within the microstate shape. For a square microstate we
have $\sigma = N^2$, and for a triangular microstate $\sigma = N(N - 1)\div 2$. In our example, it is simple since
our shape is fixed: $\sigma = 5$, so:
```@example dev
RecurrenceMicrostatesAnalysis.get_histogram_size(::MyMicrostateShape) = 2^5
```

The next step is to define our power vector. It determines how we read our microstate as an integer. Each position
should be associated with a power of 2:
```math
\begin{matrix}
2^0r_{(i,j)} & & 2^1r_{(i, j+2)} \\
 & 2^2r_{(i+1, j+1)} & \\ 
 2^3r_{(i+2,j)} & & 2^4r_{(i+2, j+2)}
\end{matrix}
```

Then, we write:
```@example dev
RecurrenceMicrostatesAnalysis.get_power_vector(::RecurrenceMicrostatesAnalysis.CPUCore, ::MyMicrostateShape) = SVector{5, Int}([1, 2, 4, 8, 16])
```

Finally, we need to define the set of offsets used to construct the microstate from the trajectory.
Note that each offset must have the same index as the corresponding element in the power vector.
```@example dev
function RecurrenceMicrostatesAnalysis.get_offsets(::RecurrenceMicrostatesAnalysis.CPUCore, ::MyMicrostateShape)
    elems = [
        SVector{2, Int}([0, 0]),
        SVector{2, Int}([0, 2]),
        SVector{2, Int}([1, 1]),
        SVector{2, Int}([2, 0]),
        SVector{2, Int}([2, 2])
    ]

    return SVector{5, SVector{2, Int}}(elems)
end
```

Now, we just need to define how our microstate will behave with respect to a sampling space. It is not necessary to define it
for all available sampling spaces, but you need to do so for at least one of them.
```@example dev
RecurrenceMicrostatesAnalysis.SamplingSpace(
    ::MyMicrostateShape, 
    x::StateSpaceSet, 
    y::StateSpaceSet
) = RecurrenceMicrostatesAnalysis.SSRect2(length(x) - 2, length(y) - 2)
```

And that's it, we can now use our new microstate shape 🙂 (and why not combine it with the previous recurrence expression?!)
```@example dev
rmspace = RecurrenceMicrostates(MyRecurrenceExpr(0.27), MyMicrostateShape())
```

```@example dev
probabilities(rmspace, X)
```

## Adding a new quantity estimator

Since **RecurrenceMicrostatesAnalysis.jl** uses the same structure as **ComplexityMeasures.jl**
to estimate or measure complexity values (e.g., determinism, disorder, etc.), the method to implement new
features is very similar.

To add new quantity estimators, refer to the [ComplexityMeasures.jl Dev Docs](https://juliadynamics.github.io/DynamicalSystemsDocs.jl/complexitymeasures/stable/devdocs/).

Tests for new quantifiers implemented on **RecurrenceMicrostatesAnalysis.jl** need to be add to `test/rqa.jl`.

## Adding a new GPU metric

Due to the incompatibility of **Distances.jl** with GPUs, it may be necessary to redefine some metrics
to use them with the **RecurrenceMicrostatesAnalysis.jl** GPU backend.

### Steps

1. Define a new type that is a subtype of [`GPUMetric`](@ref).
2. Implement the dispatch `gpu_evaluate(::YourMetric, x, y, i, j, n)`. Here, `i` and `j` indicate which
    positions of the `AbstractGPUVector` are accessed (`i` for `x`, and `j` for `y`), and `n` is the number
    of dimensions of the system.
3. Add a docstring to your metric describing it.
4. Add your metric to `docs/src/api.md`.
5. Add the expression to the [`GPUMetric`](@ref) docstring.
6. Add tests to `test/utils.jl`.