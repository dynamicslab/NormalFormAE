function rhs(dx,x,p,t)
    dx[1] = p[1]*x[1]-x[2]-x[1]*(x[1]^2+x[2]^2)
    dx[2] = x[1]+p[1]*x[2]-x[2]*(x[1]^2+x[2]^2)
    return dx
end

function rhs_hopf(x,t,p)
    dx = zeros(2,1)
    rhs(dx,x,p,t)
    return dx
end

function rhs(t,x::Array{T,2},p) where {T<:Number}
    return hcat([rhs_hopf(x[:,i],t[i],p) for i in 1:size(x)[2]]...)
end

# function rhs_GPU(du,u,p,t)
#     dx = Zygote.Buffer(du,size(du)[1])
#     for i in 1:args["nBatches"]
#         index = (i-1)*args["z_dim"]
#         dx[index+1:index+args["z_dim"]] = rhs(dx[index+1:index+args["z_dim"]],u[index+1:index+args["z_dim"]],p,t)
#     end
#     du .= copy(dx)
#     nothing
# end
# end

function rhs_solve(du,u,p,t)
    dx = Zygote.Buffer(du,args["z_dim"])
    dx[1] = p[1]*u[1]-u[2]-u[1]*(u[1]^2+u[2]^2)
    dx[2] = u[1]+p[1]*u[2]-u[2]*(u[1]^2+u[2]^2)
    du .= copy(dx)
    nothing
end



function rhs(x,p)
    return [p[1,:]'.*x[1,:]'.-x[2,:]'.-x[1,:]'.*(x[1,:]'.^2 .+x[2,:]'.^2); x[1,:]'.+p[1,:]'.*x[2,:]'.-x[2,:]'.*(x[1,:]'.^2 .+x[2,:]'.^2)]
end
