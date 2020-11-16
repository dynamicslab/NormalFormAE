# dxdt_rhs
function dxdt_solve(args,dx_,u,par,t)
    Nx = args["x_dim"]/2
    S = 1
    period=12
    dx = period/Nx
    x_space = dx .* Array(-Nx/2:(Nx/2-1))
    x = u .+ args["bif_x"]
    p = par .+ args["bif_p"]
    ft_conn = args["ft_conn"]
    beta = args["beta"]
    theta = args["theta"]
    gain = args["gain"]
    tau = args["tau"]
    # II = args["II"]
    N=Int(size(x,1)/2)
    u_=x[1:N]
    II = p[1] .* exp.(-(x_space./sig).^2)
    a_=x[N+1:2*N]
    fu=1 ./ (1.0f0 .+ exp.( -beta .* (u_ .- theta)))
    psi=real(ifft(ft_conn .* fft(fu)))
    dx_ .= [(-u_ .- (gain .* a_) .+ psi .+ II); ((-a_ .+ u_)./tau)]
end

function dxdt_rhs(args,u,par,t)
    Nx = args["x_dim"]/2
    S = 1
    period=12
    dx = period/Nx
    x_space = dx .* Array(-Nx/2:(Nx/2-1))
    x = u .+ args["bif_x"]
    p = par .+ args["bif_p"]
    ft_conn = args["ft_conn"]
    beta = args["beta"]
    theta = args["theta"]
    gain = args["gain"]
    tau = args["tau"]
    # II = args["II"]
    N=Int(size(x,1)/2)
    u_=x[1:N,:]
    II = p[1,:] .* exp.(-(x_space./sig).^2)
    a_=x[N+1:2*N,:]
    fu=1 ./ (1.0f0 .+ exp.( -beta .* (u_ .- theta)))
    psi=real(ifft(ft_conn .* fft(fu,2),2))
    dx_ = [(-u_ .- (gain .* a_) .+ psi .+ II); ((-a_ .+ u_)./tau)]
    return dx_
end

# dxdt_jac 
function dxdt_jac(u,par,t)
    return u
end

function dxdt_sens(u,par,t)
    return u
end

function dxdt_sens_rhs(x,p,t,dxda)
    return x
end

# ------------------------------------------------------------------------------------

# dzdt_rhs
function dzdt_solve(dz,z,p,t)
    dz[1] = 1/p[2]^2*(p[1]*z[1]-z[2]-z[1]*(z[1]^2+z[2]^2))
    dz[2] = 1/p[2]^2*(z[1]+p[1]*z[2]-z[2]*(z[1]^2+z[2]^2))
    return dz
end

function dzdt_rhs(x,p)
    dx_ = Zygote.Buffer(x,size(x))
    dx_[1,:] = (1 ./ p[2,:]).^2 .* (p[1,:].*x[1,:] .- x[2,:] .- x[1,:] .* (x[1,:].^2 .+ x[2,:].^2))
    dx_[2,:] = (1 ./ p[2,:]).^2 .* (x[1,:] .+ p[1,:] .* x[2,:] .- x[2,:] .*(x[1,:].^2 .+ x[2,:].^2))
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
