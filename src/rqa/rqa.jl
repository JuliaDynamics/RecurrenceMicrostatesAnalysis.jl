export rma

##########################################################################################
#   Include typicall RQA estimators (complexity measures)
##########################################################################################
include("rr.jl")
include("d.jl")
include("det.jl")
include("lam.jl")
include("tt.jl")
include("rt.jl")
include("disorder.jl")
##########################################################################################
"""
    rma(ε::Float64, [x]; kwargs...) → Dict{Symbol, Float64}

Calculate all RMA estimations for a threshold `ε` and a time series `x`.
All values are estimated using \$3\\times 3\$ square microstates and a
[`Full`](@ref) sampling mode. It always uses \$\\ell_{min}=2\$, and the
microstates are retrieved restricted to half of the RP (using [`TriangleSamplingSpace`](@ref)).

## Return
The returned value constains the following entries,
which can be retrieved as a dictionary (e.g. `results[:RR]`, etc.):

* `:RR`: recurrence rate (see [`RecurrenceRate`](@ref))
* `:DET`: determinism (see [`RecurrenceDeterminism`](@ref))
* `:D`: average diagonal length (see [`RecurrenceAverageDiagonal`](@ref))
* `:LAM`: laminarity (see [`RecurrenceLaminarity`](@ref))
* `:TT`: trapping time (see [`RecurrenceTrappingTime`](@ref))
* `:MRT`: mean recurrence time (see [`RecurrenceMeanTime`][@ref])
* `:DISREM`: disorder (see [`Disorder`](@ref))
* `:RME`: recurrence microstates entropy.

All the parameters returned by `rma` are `Float64` numbers.

## Keyword arguments
- `metric::Metric`: The metric used to compute recurrence.
"""
function rma(
        ε::Float64,
        x::Union{<: StateSpaceSet{D}, <:AbstractGPUVector{<: SVector{D}}};
        metric::Metric = DEFAULT_METRIC,
    ) where {D}

    #   Compute a recurrence distribution (3 × 3)
    rmspace = RecurrenceMicrostates(ε, 3; metric = metric, sampling = Full(), space = TriangleSamplingSpace())
    probs = probabilities(rmspace, x)

    #   Disorder normalization factor
    A = _norm_factor(Val(3), Val(D))

    #   Compute quantifiers
    S = entropy(Shannon(), probs)
    rr = measure(RecurrenceRate(ε, 3), probs)
    det = measure(RecurrenceDeterminism(ε), rmspace, probs)
    d = measure(RecurrenceAverageDiagonal(2, ε), rmspace, probs)
    lam = measure(RecurrenceLaminarity(ε), rmspace, probs)
    tt = measure(RecurrenceTrappingTime(2, ε), rmspace, probs)
    rt =  measure(RecurrenceMeanTime(2, ε), rmspace, probs)
    Ξ = measure(PartialDisorder(rmspace, 3), A, probs)

    #   Construct a dict
    dict = Dict{Symbol, Float64}(
        :RR        => rr,
        :DET       => det,
        :D         => d,
        :LAM       => lam,
        :TT        => tt,
        :MRT       => rt,
        :RME       => S,
        :DISREM    => Ξ
    )

    return dict
end
##########################################################################################