export RecurrenceTrappingTime

struct RecurrenceTrappingTime{L, M, SM} <: ComplexityEstimator 
    ε::Real
    metric::M
    sampling::SM
end

function complexity(
        c::RecurrenceTrappingTime{L}, 
        x::Union{StateSpaceSet, Vector{<:Real}, <:AbstractGPUVector{<:SVector}}, 
        y::Union{StateSpaceSet, Vector{<:Real}, <:AbstractGPUVector{<:SVector}} = x;
    ) where {L}

    @assert 1 ≤ L ≤ 2 "The minimum diagonal length must be 1 or 2."

    rmspace = RecurrenceMicrostates(c.ε, RectMicrostate(1, L + 1); metric = c.metric, sampling = c.sampling, space = TriangleSamplingSpace())
    probs = probabilities(rmspace, x, y)
    return measure(c, rmspace, probs)
end

# -- Constructors
RecurrenceTrappingTime(Lmin::Int, ε::Real; metric::M = DEFAULT_METRIC, ratio::Float64 = 0.1, sampling::SM = SRandom(ratio)) where {M,SM} = RecurrenceTrappingTime{Lmin,M,SM}(ε, metric, sampling)

##########################################################################################
#   Internal: measure from probabilities
##########################################################################################
function measure(_::RecurrenceTrappingTime{1}, rmspace::RecurrenceMicrostates, probs::Probabilities)
    if (rmspace.shape isa Rect2Microstate{2, 2, 2} && length(probs) == 16)
        rr = measure(RecurrenceRate(0.0), probs)
        div_val = probs[2] + probs[6] + probs[10] + probs[14]
        return rr / div_val
    elseif (rmspace.shape isa Rect2Microstate{1, 2, 2} && length(probs) == 4)
        return 1 + (probs[4] / probs[2])
    else
        msg = "Trapping time must be computed using square or row microstates with n = 2 for lmin = 1."
        throw(ArgumentError(msg))
    end
end

function measure(_::RecurrenceTrappingTime{2}, rmspace::RecurrenceMicrostates, probs::Probabilities)
    if (rmspace.shape isa Rect2Microstate{3, 3, 2} && length(probs) == 512)
        num_val = 0.0
        div_val = 0.0

        for a1 ∈ 0:1, a2 ∈ 0:1, a3 ∈ 0:1, a4 ∈ 0:1, a5 ∈ 0:1, a6 ∈ 0:1, a7 ∈ 0:1
            I_num = 1 + (1 + 2 + a1 * 4 + a2 * 8 + a3 * 16 + a4 * 32 + a5 * 64 + a6 * 128 + a7 * 256)
            num_val += probs[I_num]

            if (a1 == 1)
                I_div = 1 + (2 + 4 + a2 * 8 + a3 * 16 + a4 * 32 + a5 * 64 + a6 * 128 + a7 * 256)
                div_val += probs[I_div]
            end
        end

        return 1 + (num_val / div_val)
    elseif (rmspace.shape isa Rect2Microstate{1, 3, 2} && length(probs) == 8)
        return 1 + ((probs[4] + probs[8]) / probs[7])
    else
        msg = "Trapping time must be computed using square or row microstates with n = 3 for lmin = 2."
        throw(ArgumentError(msg))
    end
end