function reconstruct(nfae,x_test,alpha_test,lift,drop)
    z = cpu(nfae.state.encoder)(cpu(lift*x_test))
    init_z = z[:,1]

    beta = cpu(nfae.par.encoder)([cpu(alpha_test)])

    prob = ODEProblem(nfae.model_z.solve,
                            init_z,
                            nfae.model_x.tspan,
                            [par_[1],nfae.tscale[1]])
    t = range(0,nfae.model_x.tspan[2], length = nfae.model_x.tsize)
    sol = solve(prob,Tsit5(),saveat=t,
                dt_max=(nfae.model_x.tspan[2])/nfae.model_x.tize,
                reltol=1e-8,abstol=1e-8)
    x_ = Array(sol)

    x_back = cpu(nfae.state.decoder)(x_)

FileIO.save("/home/kaliam/NFAEdata/fluid-Hopf/x_back.jld2","x_back",x_back)
end
