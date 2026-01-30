```@meta
EditURL = "tutorial.jl"
```

````@example tutorial
using Distributions, RecurrenceMicrostatesAnalysis
data = rand(Uniform(0, 1), 10_000);
ssset = StateSpaceSet(data)

ε = 0.27
N = 2
dist = distribution(ssset, ε, N)
````

