# x: Hopf NF + quartic nonlinearity
# z: Hopf NF
# par: scalar Hopf parameter; subcritical Hopf

# dxdt_rhs
function dxdt_rhs(dx,x,p,t)
    dx[1] = p[1]*x[1]-x[2]-x[1]*(x[1]^2+x[2]^2) + x[2]^4
    dx[2] = x[1]+p[1]*x[2]-x[2]*(x[1]^2+x[2]^2) + x[1]^4
    return dx
end

function dxdt_rhs(x,p,t)
    dx_ = Zygote.Buffer(x,args["x_dim"],args["tsize"])
    dx_[1,:] = p[1].*x[1,:] .- x[2,:] .- x[1,:] .* (x[1,:].^2 .+ x[2,:].^2) .+ x[2,:].^4
    dx_[2,:] = x[1,:] .+ p[1] .* x[2,:] .- x[2,:] .*(x[1,:].^2 .+ x[2,:].^2) .+ x[1,:].^4
    return copy(dx_)
end

# dxdt_jac 
function dxdt_jac(x,p,t)
    dx_ = Zygote.Buffer(x,args["x_dim"],args["x_dim"])
    dx_[1,1] = p[1]-(3.0f0*x[1]^2+x[2]^2)
    dx_[1,2] = -1.0f0-x[1]*(2.0f0*x[2]) + 4.0f0*x[2]^3
    dx_[2,1] = 1.0f0-x[2]*(2.0f0*x[1]) + 4.0f0*x[1]^3
    dx_[2,2] = p[1]-(x[1]^2+3.0f0*x[2]^2) 
    return copy(dx_)
end

function dxdt_sens(x,p,t)
    dx_ = Zygote.Buffer(x,args["x_dim"],args["par_dim"])
    dx_[1,1] = x[1]
    dx_[2,1] = x[2]
    return copy(dx_)
end

function dxdt_sens_rhs(x,p,t,dxda)
    return dxdt_jac(x,p,t)*dxda .+ dxdt_sens(x,p,t)
end

# ------------------------------------------------------------------------------------

# dzdt_rhs
function dzdt_rhs(dz,z,p,t)
    dz[1] = p[1]*z[1]-z[2]-z[1]*(z[1]^2+z[2]^2)
    dz[2] = z[1]+p[1]*z[2]-z[2]*(z[1]^2+z[2]^2)
    return dz
end

function dzdt_rhs(x,p,t)
    dx_ = Zygote.Buffer(x,args["x_dim"],args["tsize"])
    dx_[1,:] = p[1].*x[1,:] .- x[2,:] .- x[1,:] .* (x[1,:].^2 .+ x[2,:].^2)
    dx_[2,:] = x[1,:] .+ p[1] .* x[2,:] .- x[2,:] .*(x[1,:].^2 .+ x[2,:].^2)
    return copy(dx_)
end

# dzdt_jac 
function dzdt_jac(x,p,t)
    dx_ = Zygote.Buffer(x,args["z_dim"],args["z_dim"])
    dx_[1,1] = p[1]-(3.0f0*x[1]^2+x[2]^2)
    dx_[1,2] = -1.0f0-x[1]*(2.0f0*x[2])
    dx_[2,1] = 1.0f0-x[2]*(2.0f0*x[1])
    dx_[2,2] = p[1]-(x[1]^2+3.0f0*x[2]^2)
    return copy(dx_)
end

function dzdt_sens(x,p,t)
    dx_ = Zygote.Buffer(x,args["z_dim"],args["par_dim"])
    dx_[1,1] = x[1]
    dx_[2,1] = x[2]
    return copy(dx_)
end

function dzdt_sens_rhs(x,p,t,dxda)
    return dxdt_jac(x,p,t)*dxda .+ dxdt_sens(x,p,t)
end 
