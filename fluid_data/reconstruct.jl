function reconstruct(nfae,x_test,alpha_test,lift,drop)
    z = cpu(nfae.state.encoder)(cpu(drop*x_test))
    init_z = z[:,1]

    beta = cpu(nfae.par.encoder)([cpu(alpha_test)])

    prob = ODEProblem(nfae.model_z.solve,
                            init_z,
                            nfae.model_x.tspan,
                            [beta[1],nfae.tscale[1]])
    t = range(0,nfae.model_x.tspan[2], length = nfae.model_x.tsize)
    sol = solve(prob,Tsit5(),saveat=t,
                dt_max=(nfae.model_x.tspan[2])/nfae.model_x.tsize,
                reltol=1e-8,abstol=1e-8)
    x_ = Array(sol)

    x_back = lift*cpu(nfae.state.decoder)(x_)

    FileIO.save("$(pwd())/NFAEdata/fluid-Hopf/x_back.jld2","x_back",x_back)
    return x_back
end

function gen_data_fluid(nfae,x_test,alpha_test,lift,drop,test_ind,ind)
    x_test = nfae.test_data["x"][:,:,ind]
    x_back_1 = reconstruct(nfae,x_test,alpha_test[ind],lift,drop)
    x_back_2 = gpu(lift)*nfae.state.decoder(nfae.state.encoder(drop*x_test))
    FileIO.save("$(pwd())/fluid_data/x_back_1.jld2","x_back_1",cpu(x_back_1))
    FileIO.save("$(pwd())/fluid_data/x_back_2.jld2","x_back_2",cpu(x_back_2))
    FileIO.save("$(pwd())/fluid_data/x_gt.jld2","x_gt",cpu(x_test))
    x_mode = FileIO.load("$(pwd())/NFAEdata/fluid-Hopf/fluid_data_full.jld2","fluid_data")["xmode"][:,:,test_ind[ind]]
    FileIO.save("$(pwd())/fluid_data/x_mode.jld2","x_mode",cpu(x_mode))
end

function load_data_fluid()
    x_back_1 = FileIO.load("$(pwd())/fluid_data/x_back_1.jld2","x_back_1")
    x_back_2 = FileIO.load("$(pwd())/fluid_data/x_back_2.jld2","x_back_2")
    x_mode = FileIO.load("$(pwd())/fluid_data/x_mode.jld2","x_mode")
    x_gt = FileIO.load("$(pwd())/fluid_data/x_gt.jld2","x_gt")
    return x_back_1,x_back_2,x_mode,x_gt
end


    
function plot_fluid(v_back, v_gt, x_mode, x_mean,x_full, index,index_2)
    Re = 80; # Reynolds number

    Δx, Δt = setstepsizes(Re,gridRe=2)


    tfinal = 3*Δt
    # tsize = Int(div(tfinal,Δt)) + 2
    x_size = 486
    y_size = 250
    vec_size = x_size*y_size
    svd_index = 325
    nmodes = 10

    Re = 80

        ## STEP 1: Simulate Navier Stokes over high fidelity grid
        U = 1.0; # Free stream velocity
        U∞ = (U,0.0);
        xlim = (-2.0,10.0)
        ylim = (-3.0,3.0);
        body = Circle(0.5,Δx)
        sys = NavierStokes(Re,Δx,xlim,ylim,Δt,body,freestream = U∞)
        u0 = newstate(sys);
        tspan = (0.0,tfinal)
        tt = range(0.0,tfinal,length=1000)
        integrator = init(u0,tspan,sys)
        @time step!(integrator,tfinal)
        
    reconstructed_xx = x_mode*v_back .+ x_mean;
    original_xx = x_mode*v_gt .+ x_mean
    original_xx = x_full

    for i in 1:121750
    integrator.u[i] = reconstructed_xx[i,index]
    end

    min_ = minimum(vorticity(integrator))
    max_ = maximum(vorticity(integrator))

    plot(vorticity(integrator),sys,title="Vorticity",clim=(0.25*min_,0.25*max_),levels=range(0.5*min_,0.5*max_,length=50), color = :RdBu)
    plot!(body)
    savefig("$(pwd())/NFAEdata/fluid-Hopf/reconstructed.pdf")

    min_ = minimum(vorticity(integrator))
    max_ = maximum(vorticity(integrator))
    
    for i in 1:121750
    integrator.u[i] = original_xx[i,end]
    end

    #min_ = minimum(vorticity(integrator))
    # max_ = maximum(vorticity(integrator))

    plot(vorticity(integrator),sys,title="Vorticity",clim=(0.25*min_,0.25*max_),levels=range(0.5*min_,0.5*max_,length=50), color = :RdBu)
    plot!(body)
    savefig("$(pwd())/NFAEdata/fluid-Hopf/original.pdf")

    plot(reconstructed_xx[index_2,:],label="Reconstructed")
    plot!(original_xx[index_2,:],label="Original")

    for j in 1:5
        for i in 1:121750
            integrator.u[i] = original_xx[i,end-10*(j-1)]
        end

        #min_ = minimum(vorticity(integrator))
        # max_ = maximum(vorticity(integrator))

        plot(vorticity(integrator),sys,title="Vorticity",clim=(0.25*min_,0.25*max_),levels=range(0.5*min_,0.5*max_,length=50), color = :RdBu)
        plot!(body)
        savefig("$(pwd())/NFAEdata/fluid-Hopf/sol_$(6-j).pdf")
    end

    
    savefig("$(pwd())/NFAEdata/fluid-Hopf/timetrace.pdf")
    return sys
end



function plot_modes(x_full)
    Re = 80; # Reynolds number

    Δx, Δt = setstepsizes(Re,gridRe=2)


    tfinal = 3*Δt
    # tsize = Int(div(tfinal,Δt)) + 2
    x_size = 486
    y_size = 250
    vec_size = x_size*y_size
    svd_index = 325
    nmodes = 10

    Re = 80

        ## STEP 1: Simulate Navier Stokes over high fidelity grid
        U = 1.0; # Free stream velocity
        U∞ = (U,0.0);
        xlim = (-2.0,10.0)
        ylim = (-3.0,3.0);
        body = Circle(0.5,Δx)
        sys = NavierStokes(Re,Δx,xlim,ylim,Δt,body,freestream = U∞)
        u0 = newstate(sys);
        tspan = (0.0,tfinal)
        tt = range(0.0,tfinal,length=1000)
        integrator = init(u0,tspan,sys)
        @time step!(integrator,tfinal)
        
    
    x_size = 486
    y_size = 250
    vec_size = x_size*y_size
    svd_index = 325
    
    vec_ = x_full    
    vec_dim = size(vec_)

    ## STEP 3: SVD

    tmp = vec_[:,svd_index];
    for i = (svd_index+1):vec_dim[2]   
        tmp = tmp .+ vec_[:,i]
    end
    vec_avg = tmp./(vec_dim[2]-svd_index + 1)
    vec_svd = vec_[:,svd_index:end] .- vec_avg
    s = svd(vec_svd)
    
    sing_vals = s.S
    plot(sing_vals,yaxis=:log)
    savefig("$(pwd())/NFAEdata/fluid-Hopf/sing_vals.pdf")
   
    min_ = minimum(vorticity(integrator))
    max_ = maximum(vorticity(integrator))

    for i in 1:6
    
        for j in 1:121750
            integrator.u[j] = s.U[j,i]
        end
       
        if i == 1
            min_ = minimum(vorticity(integrator))
            max_ = maximum(vorticity(integrator))
        end
        plot(vorticity(integrator),sys,title="Mode $(i)",clim=(min_,max_),levels=range(min_,max_,length=50), color = :RdBu)
        plot!(body)
        savefig("$(pwd())/NFAEdata/fluid-Hopf/mode_$(i).pdf")
        
    end
    
    for i in 1:2:5

        plot(s.S[i] .* s.Vt[i,:],ylim=(-2.5,2.5))
        plot!(s.S[i+1] .* s.Vt[i+1,:],ylim=(-2.5,2.5))
        savefig("$(pwd())/NFAEdata/fluid-Hopf/tmode_$(i).pdf")
        
    end


return sys
end
