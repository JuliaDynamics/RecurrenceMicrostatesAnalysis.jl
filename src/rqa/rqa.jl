export rma

##########################################################################################
#   Include typicall RQA estimators (complexity measures)
##########################################################################################
include("rr.jl")
include("det.jl")
include("lam.jl")
include("disorder.jl")
##########################################################################################
"""
    rma(ε::Float64, [x]; kwargs...) → Dict{Symbol, Float64}

Calculate all RMA estimations for a threshold `ε` and a time series `x`.
All values are estimated using \$3\\times 3\$ square microstates and a
[`Full`](@ref) sampling mode.

## Return
The returned value constains the following entries,
which can be retrieved as a dictionary (e.g. `results[:RR]`, etc.):

* `:RR`: recurrence rate (see [`RecurrenceRate`](@ref))
* `:DET`: determinism (see [`RecurrenceDeterminism`](@ref))
* `:LAM`: laminarity (see [`RecurrenceLaminarity`](@ref))
* `:DISREM`: disorder (see [`Disorder`](@ref))
* `:RENT`: recurrence entropy.

All the parameters returned by `rma` are `Float64` numbers.

## Keyword arguments
- `metric::Metric`: The metric used to compute recurrence.
"""
function rma(
        ε::Float64,
        x::StateSpaceSet;
        metric::Metric = DEFAULT_METRIC,
    )

    #   Compute a recurrence distribution (3 × 3)
    rmspace = RecurrenceMicrostates(ε, 3; metric = metric, sampling = Full())
    probs = probabilities(rmspace, x)

    #   Compute quantifiers
    S = entropy(Shannon(), probs)
    rr = measure(RecurrenceRate(ε, 3), probs)
    det = measure(RecurrenceDeterminism(ε), rmspace, probs)
    lam = measure(RecurrenceLaminarity(ε), rmspace, probs)
    Ξ = complexity(PartialDisorder{3}(compute_labels(3), rmspace), x)

    #   Construct a dict
    dict = Dict{Symbol, Float64}(
        :RR        => rr,
        :DET       => det,
        :LAM       => lam,
        :RENT      => S,
        :DISREM    => Ξ
    )

    return dict
end
##########################################################################################