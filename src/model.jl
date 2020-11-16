using DifferentialEquations, Distributions

function data_gen(x_dim, par_dim, tsize, tspan, xVar, aVar, mean_ic_x, mean_ic_a, solver, dxdt_rhs,n_ics)

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
        prob = ODEProblem(solver,
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
        dxdt[:,:,i] = dxdt_rhs(x_,ic[x_dim+1:end,i],t)
        alpha[:,i] = ic[x_dim+1:end,i]
    end

    println("Data generated...")
    return x, dxdt, alpha
end


mutable struct xModel{x_name}
    x_dim :: Int64
    par_dim :: Int64
    tsize :: Int64
    tspan :: Array{Float64,1}
    xVar :: Float64
    aVar :: Float64
    mean_ic_x :: Array
    mean_ic_a :: Array{Float64,1}
    rhs :: Any
    solve :: Any
    gen :: Any
end

function xModel(name, x_dim, par_dim, tsize, tspan, xVar, aVar, mean_ic_x, mean_ic_a, rhs_, solve_,args)

    x_solve(dx,u,par,t) = solve_(args,dx,u,par,t)
    x_rhs(u,par,t) = rhs_(args, u, par, t)
    gen(n_ics) = data_gen(x_dim, par_dim, tsize, tspan, xVar, aVar, mean_ic_x, mean_ic_a, x_solve, x_rhs,n_ics)
    
    xModel{name}(x_dim, par_dim, tsize, tspan, xVar, aVar, mean_ic_x, mean_ic_a, x_rhs, x_solve, gen)
end

Base.show(io::IO, model::xModel{name_}) where name_  = print(io, "Model: ", String(name_),
                                                             "\nState dimension:", model.x_dim,
                                                             "\nParameter dimension:", model.par_dim)
mutable struct NormalForm{z_name}
    z_dim :: Int64
    par_dim :: Int64
    rhs :: Any
    solve :: Any    
end

function NormalForm(name, z_dim, par_dim, rhs, solve)
    NormalForm{name}(z_dim, par_dim, rhs, solve)
end

Base.show(io::IO, model::NormalForm{name_}) where name_  = print(io, "Normal Form: ", String(name_),
                                                             "\nState dimension:", model.z_dim,
                                                             "\nParameter dimension:", model.par_dim)

    


