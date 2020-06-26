# function rhs(dx,x,p,t)
#     dx[1] = p[1]*(x[2]-x[1])
#     dx[2] = x[1]*(p[2]-x[3])-x[2]
#     dx[3] = x[1]*x[2]-p[3]*x[3]
# end

function rhs(dx,x,p,t)
    dx[1] = p[1]*(x[2]-x[1])
    dx[2] = x[1]*(p[2]-x[3])-x[2]
    dx[3] = x[1]*x[2]-2.666f0*x[3]
end

function rhs_lorenz(x,t,p)
    x1,x2,x3 = x
    p1,p2 = p
    dx = [p1*(x2-x1),
          x1*(p2-x3)-x2,
          x1*x2-2.666f0*x3] |> gpu
    return dx
end

function rhs_scaled(dx,x,p,t)
    # println(typeof(x))
    # println(typeof(p))
    # println(typeof(rhs_lorenz(args["normalize"].*x,t,args["p_normalize"].*p) ))
    dx .= 1.0f0/args["normalize"].*rhs_lorenz(args["normalize"].*x,t,args["p_normalize"].*p) 
end

function rhs(t,x::Array{T,2},p) where {T<:Number}
    return hcat([rhs_lorenz(x[:,i],t[i],p) for i in 1:size(x)[2]]...)
    #return [x[3,:]'.*x[1,:]'.-x[2,:]'.+x[1,:]'.*(x[1,:]'.^2 .+x[2,:]'.^2); x[1,:]'.+x[3,:]'.*x[2,:]'.+x[2,:]'.*(x[1,:]'.^2 .+ x[2,:]'.^2); zeros(size(x)[2])']
end

# function rhs(x,p)
#     return (1/args["normalize"]).*[(args["p_normalize"].*p[1,:]').*((args["normalize"].*x[2,:]').-(args["normalize"].*x[1,:]'));(args["normalize"].*x[1,:]').*((args["p_normalize"].*p[2,:]').-(args["normalize"].*x[3,:]')).-(args["normalize"].*x[2,:]');(args["normalize"].*x[1,:]').*(args["normalize"].*x[2,:]').-(args["p_normalize"].*p[3,:]').*(args["normalize"].*x[3,:]')]
# end

function rhs(x,p)
    return (1/args["normalize"]).*[(args["p_normalize"].*p[1,:]').*((args["normalize"].*x[2,:]').-(args["normalize"].*x[1,:]'));(args["normalize"].*x[1,:]').*((args["p_normalize"].*p[2,:]').-(args["normalize"].*x[3,:]')).-(args["normalize"].*x[2,:]');(args["normalize"].*x[1,:]').*(args["normalize"].*x[2,:]').-(2.666f0).*(args["normalize"].*x[3,:]')]
end
