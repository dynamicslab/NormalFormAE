function rhs(dx,x,t,p)
    dx[1] = x[4]*(x[2]-x[1])
    dx[2] = x[1]*(x[5]-x[3])-x[2]
    dx[3] = x[1]*x[2]-x[6]*x[3]
    dx[4] = 0.0
    dx[5] = 0.0
    dx[6] = 0.0
    return dx
end

function rhs_lorenz(x,t)
    dx = zeros(6,1)
    dx .= rhs(dx,x,t,0)
    return dx
end


function rhs(t,x::Array{T,2}) where {T<:Number}
    return hcat([rhs_lorenz(x[:,i],t[i]) for i in 1:size(x)[2]]...)
    #return [x[3,:]'.*x[1,:]'.-x[2,:]'.+x[1,:]'.*(x[1,:]'.^2 .+x[2,:]'.^2); x[1,:]'.+x[3,:]'.*x[2,:]'.+x[2,:]'.*(x[1,:]'.^2 .+ x[2,:]'.^2); zeros(size(x)[2])']
end

function rhs(x)
    return [x[4,:]'.*(x[2,:]'.-x[1,:]');x[1,:]'.*(x[5,:]'.-x[3,:]').-x[2,:]';x[1,:]'.*x[2,:]'.-x[6,:]'.*x[3,:]';0.0f0 .* x[1,:]';0.0f0 .* x[1,:]';0.0f0 .* x[1,:]']
end
