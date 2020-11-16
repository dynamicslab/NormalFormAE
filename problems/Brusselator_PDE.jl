# x: Hopf NF + quartic nonlinearity
# z: Hopf NF
# par: scalar Hopf parameter; subcritical Hopf

# dxdt_rhs
function dxdt_rhs(dx_,u,par,t)
    x = u .+ args["bif_x"]
    p = par .+ args["bif_p"]
    aa = args["par_aa"]
    bb = args["par_bb"]
    dx = args["par_dx"]
    dy = args["par_dy"]
    # x = 0
    dx_[1] = 0.0f0
    dx_[2] = 0.0f0
    # x = N
    dx_[end-1] = 0.0f0
    dx_[end] = 0.0f0
    i = 3
    while i<(args["x_dim"]-2)
        dx_[i] = (dx/p[1]*(x[i-2] + x[i+2] - 2.0f0 * x[i])*(args["x_dim"]-1)^2 + x[i]^2*x[i+1] - (bb + 1.0f0)*x[i] + aa)
        dx_[i+1] = (dy/p[1]*(x[i-2] + x[i+2] -2.0f0 * x[i])*(args["x_dim"]-1)^2 + bb*x[i-1] - x[i-1]^2*x[i])
        i = i+2
    end
    return dx
end

function dxdt_rhs(u,par,t)
    x = u .+ args["bif_x"]
    p = par .+ args["bif_p"]
    dx_ = Zygote.Buffer(x,args["x_dim"],args["tsize"])
    ind_ = args["x_dim"]
    aa = args["par_aa"]
    bb = args["par_bb"]
    dx = args["par_dx"]
    dy = args["par_dy"]
    # x = 0
    dx_[1,:] = zeros(Float32,args["tsize"])
    dx_[2,:] = zeros(Float32,args["tsize"])
    # x = N
    dx_[ind_-1,:] = zeros(Float32,args["tsize"])
    dx_[ind_,:] = zeros(Float32,args["tsize"])
    i = 3
    while i<(args["x_dim"]-2)
        dx_[i,:] = dx/p[1] .* (x[i-2,:] .+ x[i+2,:] .- 2.0f0 .* x[i,:]) .* (args["x_dim"]-1)^2 .+ (x[i,:] .^ 2.0f0) .* x[i+1,:] .- (bb + 1.0f0) .* x[i,:] .+ aa
        dx_[i+1,:] = dy/p[1] .* (x[i-2,:] .+ x[i+2,:] .- 2.0f0 .* x[i,:]) .* (args["x_dim"]-1)^2 .+ bb .* x[i-1,:] .- (x[i-1,:].^2.0f0) .* x[i,:]
        i = i+2
    end
    return copy(dx_)
end

# dxdt_jac 
function dxdt_jac(u,par,t)
    x = u .+ args["bif_x"]
    p = par .+ args["bif_p"]
    dx_ = Zygote.Buffer(x,args["x_dim"],args["x_dim"])
    ind_ = args["x_dim"]
    dx_[1,1] = -1.0f0
    dx_[1,2] = x[ind_]
    dx_[1,3:ind_-2] = zeros(Float32,ind_-4)
    dx_[1,ind_-1] = - x[ind_]
    dx_[1,ind_] = x[2]
    # ------------------
    dx_[2,1] = x[3]
    dx_[2,2] = -1.0f0
    dx_[2,3] = x[1]
    dx_[2,4:ind_-1] = zeros(Float32,ind_-4)
    dx_[2,ind_] = -x[1]
    # -----------------
    for i=3:(size(dx_,1)-1)
        if i != 3
            dx_[i,1:i-3] = zeros(Float32,i-3)
        end
        dx_[i,i-2] = -x[i-1]
        dx_[i,i-1] = x[i+1]
        dx_[i,i] = -1.0f0
        dx_[i,i+1] = x[i-1]
        if i != args["x_dim"]
            dx_[i,i+2:ind_] = zeros(Float32,ind_-i-1)
        end
    end
    # ----------------
    dx_[ind_,1] = x[ind_-1]
    dx_[ind_,2:ind_-3] = zeros(Float32,ind_-4)
    dx_[ind_,ind_-2] = -x[ind_-1]
    dx_[ind_,ind_-1] = x[1]
    dx_[ind_,ind_] = -1.0f0
    return copy(dx_)
end

function dxdt_sens(u,par,t)
    x = u .+ args["bif_x"]
    p = par .+ args["bif_p"]
    dx_ = Zygote.Buffer(x,args["x_dim"],args["par_dim"])
    dx_[1:args["x_dim"],1] = zeros(Float32,args["x_dim"]) .+ 1.0f0
    return copy(dx_)
end

function dxdt_sens_rhs(x,p,t,dxda)
    return dxdt_jac(x,p,t)*dxda .+ dxdt_sens(x,p,t)
end

# ------------------------------------------------------------------------------------

# dzdt_rhs
function dzdt_solve(dz,z,p,t)
    dz[1] = p[1]*z[1]-z[2]-z[1]*(z[1]^2+z[2]^2)
    dz[2] = z[1]+p[1]*z[2]-z[2]*(z[1]^2+z[2]^2)
    return dz
end

function dzdt_rhs(x,p,t)
    dx_ = Zygote.Buffer(x,size(x))
    dx_[1,:] = p[1,:].*x[1,:] .- x[2,:] .- x[1,:] .* (x[1,:].^2 .+ x[2,:].^2)
    dx_[2,:] = x[1,:] .+ p[1,:] .* x[2,:] .- x[2,:] .*(x[1,:].^2 .+ x[2,:].^2)
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
    return dzdt_jac(x,p,t)*dxda .+ dzdt_sens(x,p,t)
end 
