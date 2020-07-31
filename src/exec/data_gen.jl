using DifferentialEquations, DiffEqSensitivity, Distributions, PolyChaos


function gen(args,dxdt_rhs,dxdt_sens_rhs,n_ics,type_="training")

    # Needs:
    # dxdt_rhs
    # dxdt_jac 
    # dxdt_sens 
    # dzdt_rhs
    # dzdt_jac
    # dzdt_sens

    # The above are supplied in problem/hopf.jl (as an example)    

    # mean_ic_x
    mean_ic_x = args["mean_init"]
    # mean_ic_par
    mean_ic_a = args["mean_a"]

    # Generate init
    dist_ = Uniform(-1.0,1.0)
    ic_x = mean_ic_x .+ args["xVar"].*rand(dist_,n_ics,args["x_dim"])'
    ic_a = mean_ic_a .+ args["aVar"].*rand(dist_,n_ics,args["par_dim"])'
    ic = [ic_x; ic_a]

    # Initialize data
    dxda = zeros(args["x_dim"],args["tsize"],args["par_dim"],n_ics)
    x = zeros(args["x_dim"],args["tsize"],n_ics)
    dxdt = zeros(args["x_dim"],args["tsize"],n_ics)
    dtdxda = zeros(args["x_dim"],args["tsize"],args["par_dim"],n_ics)
    alpha = zeros(args["par_dim"],n_ics)
    
    # Generate sensitivity dxda and dtdxda
    for i in 1:n_ics
        #prob = ODEForwardSensitivityProblem(dxdt_rhs, ic[1:args["x_dim"],i],(args["tspan"][1],args["tspan"][2]),ic[args["x_dim"]+1:end,i])
        prob = ODEProblem(dxdt_rhs, ic[1:args["x_dim"],i],(args["tspan"][1],args["tspan"][2]),ic[args["x_dim"]+1:end,i])
        t = range(args["tspan"][1],args["tspan"][2],length = args["tsize"])
        sol = solve(prob,BS3(),saveat=t,dt_max=(args["tspan"][2]-args["tspan"][1])/args["tsize"],reltol=1e-8,abstol=1e-8)
        #x_,dxda_ = extract_local_sensitivities(sol)
        x_ = Array(sol)
        try
            x[:,:,i] = x_
        catch e
            p = plot(x_')
            display(p)
            println(ic[:,i])
        end
        dxdt[:,:,i] = dxdt_rhs(x_,ic[args["x_dim"]+1:end,i],t)
        #dxda[:,:,:,i] = reshape(hcat(dxda_...),args["x_dim"],args["tsize"],args["par_dim"])
        #dtdxda_ = vcat([dxdt_sens_rhs(x_[:,j],ic[args["x_dim"]+1:end,i],0.0f0,dxda[:,j,:,i]) for j in 1:args["tsize"]]...)
        #dtdxda[:,:,:,i] = reshape(dtdxda_,args["x_dim"],args["tsize"],args["par_dim"])
        alpha[:,i] = ic[args["x_dim"]+1:end,i]
    end

    println("Data generated...")
    return x, dxdt, alpha, dxda, dtdxda
end

