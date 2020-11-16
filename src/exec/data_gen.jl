using DifferentialEquations, DiffEqSensitivity, Distributions, PolyChaos


function data_gen(x_dim, par_dim, tsize, tspan, xVar, aVar, mean_ic_x, mean_ic_a, solve, rhs,n_ics)

    # Generate init
    dist_ = Uniform(-1.0,1.0)
    ic_x = mean_ic_x .+ xVar.*rand(dist_,n_ics,x_dim)'
    ic_a = mean_ic_a .+ aVar.*rand(dist_,n_ics,par_dim)'
    ic = [ic_x; ic_a]

    # Initialize data
    x = zeros(x_dim,tsize,n_ics)
    dxdt = zeros(x_dim,tsize,n_ics)
    alpha = zeros(par_dim,n_ics)
    
    for i in 1:n_ics
        prob = ODEProblem(dxdt_rhs,
                          ic[1:x_dim,i],
                          (tspan[1],tspan[2]),
                          ic[x_dim+1:end,i])
        t = range(tspan[1],tspan[2], length = tsize)
        sol = solve(prob,Tsit5(),saveat=t,
                    dt_max=(tspan[2]-tspan[1])/tsize,
                    reltol=1e-8,abstol=1e-8)
        x_ = Array(sol)
        try
            x[:,:,i] = x_
        catch e
            p = plot(x_')
            display(p)
            println(i)
            println(ic[:,i])
        end
        dxdt[:,:,i] = rhs(x_,ic[x_dim+1:end,i],t)
        alpha[:,i] = ic[x_dim+1:end,i]
    end

    println("Data generated...")
    return x, dxdt, alpha
end

