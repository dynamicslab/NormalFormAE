using DifferentialEquations, Distributions, PolyChaos


function gen(args,rhsfun,n_ics, noise_strength = 0,type_="training" )
    
    # simulation
    function sim(init_val,rhs_,t,p_)
        prob_ = ODEProblem(rhs_,init_val,(t[1],t[end]),p_)
        sol = Array(solve(prob_,Tsit5(),saveat=t,dt_max = (t[end]-t[1])/args["tsize"]))
        return sol
    end
    
    # Spatial modes
    function spatial_scale(args,z,dz,spatial_scale)
        # z = n_ics x z_dim x t_steps
        z_dim = size(z)[2]
        n_modes = z_dim*args["expansion_order"]
        x_range = Array(range(-1,1,length=spatial_scale))
        op_legendre = PolyChaos.LegendreOrthoPoly(n_modes)
        modes = zeros(n_modes,spatial_scale)
        for i=1:n_modes
            modes[i,:] = PolyChaos.evaluate(i,x_range,op_legendre)
        end
        x = zeros(size(z)[1],spatial_scale,size(z)[3])
        dx = zeros(size(z)[1],spatial_scale,size(z)[3])
        for i=1:size(z)[1]
            for k in 1:size(z)[3]
                for j in 1:size(z)[2]                    
                    x[i,:,k] += modes[j,:].*z[i,j,k]
                    dx[i,:,k] += modes[j,:].*dz[i,j,k]
                    if args["expansion_order"] >= 2
                        x[i,:,k] += modes[j+z_dim,:].*z[i,j,k]^2
                        dx[i,:,k] += modes[j+z_dim,:].*2.0 .* z[i,j,k].*dz[i,j,k]                        
                    end
                    if args["expansion_order"] >= 3
                        x[i,:,k] += modes[j+2*z_dim,:].*z[i,j,k]^3
                        dx[i,:,k] += modes[j+2*z_dim,:].*3.0 .*z[i,j,k]^2 .*dz[i,j,k]
                    end
                end
            end
        end
        return x,dx
    end
    
    t = range(args["tspan"][1],args["tspan"][2],length = args["tsize"])
    x_dim = args["x_spatial_scale"]
    dist_ = Uniform(-1.0,1.0)
    mean_ic = args["mean_init"]
    ics = noise_strength.*rand(dist_,n_ics,args["z_dim"]+args["par_dim"])'.+mean_ic
    z = zeros(n_ics,args["z_dim"],args["tsize"])
    dz = zeros(n_ics,args["z_dim"],args["tsize"])
    par = zeros(n_ics,args["par_dim"],args["tsize"])
    par_temp = zeros(n_ics,args["par_dim"],args["tsize"])
    for i=1:size(ics,2)
        ic_ = ics[1:args["z_dim"],i]
        p_ = ics[args["z_dim"]+1:end,i]
        par[i,:,:] = hcat([p_ for i in 1:args["tsize"]]...)
        z[i,:,:] = sim(ic_,rhsfun,t,p_)
        dz[i,:,:] = rhsfun(t,z[i,:,:],p_)
    end
    z ./= args["normalize"]
    dz ./= args["normalize"]
    par ./=args["p_normalize"]
    state_x,state_dx = spatial_scale(args,z,dz,x_dim)
    alpha,alpha_prime = spatial_scale(args,par,par_temp,args["alpha_spatial_scale"])
    if type_ == "training"
        return t,z,dz,state_x,state_dx,par,alpha
    else
        z_ = z[1,:,:]
        for i=2:n_ics
            z_ = [z_ z[i,:,:]]
        end
        state_x_ = state_x[1,:,:]
        for i=2:n_ics
            state_x_ = [state_x_ state_x[i,:,:]]
        end
        dz_ = dz[1,:,:]
        for i=2:n_ics
            dz_ = [dz_ dz[i,:,:]]
        end
        state_dx_ = state_dx[1,:,:]
        for i=2:n_ics
            state_dx_ = [state_dx_ state_dx[i,:,:]]
        end
        par_ = par[1,:,:]
        for i=2:n_ics
            par_ = [par_ par[i,:,:]]
        end
        alpha_ = alpha[1,:,:]
        for i=2:n_ics
            alpha_ = [alpha_ alpha[i,:,:]]
        end
        return t,z_,dz_,state_x_,state_dx_,par_,alpha_
    end
end

