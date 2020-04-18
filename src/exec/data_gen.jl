using DifferentialEquations, Distributions, PolyChaos


function gen(args,rhsfun,n_ics, noise_strength = 0,type_="training" )
    
    # simulation
    function sim(init_val,rhs_,t)
        prob_ = ODEProblem(rhs_,init_val,(t[1],t[end]))
        sol = Array(solve(prob_,Tsit5(),saveat=t,dt_max = (t[end]-t[1])/args["tsize"]))
        return sol
    end
    
    # Spatial modes
    function spatial_scale(args,z,dz)
        # z = n_ics x z_dim x t_steps
        n_modes = args["z_dim"]*args["expansion_order"]
        x_range = Array(range(-1,1,length=args["spatial_scale"]))
        op_legendre = PolyChaos.LegendreOrthoPoly(n_modes)
        modes = zeros(n_modes,args["spatial_scale"])
        for i=1:n_modes
            modes[i,:] = PolyChaos.evaluate(i,x_range,op_legendre)
        end
        x = zeros(size(z)[1],args["spatial_scale"],size(z)[3])
        dx = zeros(size(z)[1],args["spatial_scale"],size(z)[3])
        for i=1:size(z)[1]
            for k in 1:size(z)[3]
                for j in 1:size(z)[2]                    
                    x[i,:,k] += modes[j,:].*z[i,j,k]
                    dx[i,:,k] += modes[j,:].*dz[i,j,k]
                    if args["expansion_order"] >= 2
                        x[i,:,k] += modes[j+args["z_dim"],:].*z[i,j,k]^2
                        dx[i,:,k] += modes[j+args["z_dim"],:].*2.0 .* z[i,j,k].*dz[i,j,k]                        
                    end
                    if args["expansion_order"] >= 3
                        x[i,:,k] += modes[j+2*args["z_dim"],:].*z[i,j,k]^3
                        dx[i,:,k] += modes[j+2*args["z_dim"],:].*3.0 .*z[i,j,k]^2 .*dz[i,j,k]
                    end
                end
            end
        end
        return x,dx
    end
    
    t = range(args["tspan"][1],args["tspan"][2],length = args["tsize"])
    x_dim = args["spatial_scale"]
    dist_ = Normal()
    mean_ic = args["mean_init"]
    ics = noise_strength.*rand(dist_,n_ics,args["z_dim"])'.+mean_ic
    z = zeros(n_ics,args["z_dim"],args["tsize"])
    dz = zeros(n_ics,args["z_dim"],args["tsize"])
    for i=1:size(ics,2)
        ic_ = ics[:,i]
        z[i,:,:] = sim(ic_,rhsfun,t)
        dz[i,:,:] = rhsfun(t,z[i,:,:])
    end
    x,dx = spatial_scale(args,z,dz)
    if type_ == "training"
        return t,z,dz,x,dx
    else
        z_ = z[1,:,:]
        for i=2:n_ics
            z_ = [z_ z[i,:,:]]
        end
        x_ = x[1,:,:]
        for i=2:n_ics
            x_ = [x_ x[i,:,:]]
        end
        dz_ = dz[1,:,:]
        for i=2:n_ics
            dz_ = [dz_ dz[i,:,:]]
        end
        dx_ = dx[1,:,:]
        for i=2:n_ics
            dx_ = [dx_ dx[i,:,:]]
        end
        return t,z_,dz_,x_,dx_
    end
end

