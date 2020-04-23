function jacobian(f, args...; matrix::Bool = false)
    res, back = Zygote.pullback(f, args...)
    if !(res isa AbstractArray)
        matrix && error("jacobian(f, args...; matrix=true) cannot " *
                        "handle scalar output, try gradient(f, args...)")
        return gradient(f, args...)
    end
    out = map(args) do p
        T = Base.promote_type(eltype(p), eltype(res))
        similar(res, T, size(res)..., size(p)...)
    end
    #delta = fill!(similar(res), 0)
    delta = Zygote.buffer(res,length(res))
    for k in CartesianIndices(res)
        delta[k] = 1
        grads = back(delta)
        for (g,o) in zip(grads, out)
            c = map(_->(:), size(g))
            o[k,c...] .= g
        end
        delta[k] = 0
    end
    matrix ? reshape.(out, length(res), :) : out
    #reshape(out[1],length(res),size(args[1])[1])
    #reshape(out[1], length(res), size(args)[1])
end

function jacobian(f, ps::Params) # Union{Tracker.Params, Zygote.Params}
    res, back = Zygote.pullback(f, ps)
    out = IdDict()
    for p in ps
        T = Base.promote_type(eltype(p), eltype(res))
        J = similar(res, T, size(res)..., size(p)...)
        out[p] = J
    end
    delta = fill!(similar(res), 0)
    for k in CartesianIndices(res)
        delta[k] = 1
        grads = back(delta)
        for p in ps
            g = grads[p]
            c = map(_->(:), size(g))
            o = out[p]
            o[k,c...] .= g
        end
        delta[k] = 0
    end
    out
end
