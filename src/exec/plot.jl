function pp(x)
    return @sprintf("%.3e",x)
end

function gen_plot(z_dim, nPlots)
    p = []
    ind_ = 1:z_dim
    if nPlots != 0
        for i in ind_
            tmp_ = plot()
            push!(p,tmp_)
        end
    end
    tmp_ = plot([0.0],[0.0],marker=2,title="Abs. Test Loss",titlefont=font(10),label="")
    push!(p,tmp_)
    tmp_ = plot([0.0],[0.0],marker=2,title="Abs. Train loss",titlefont=font(10),label="")
    push!(p,tmp_)
    tmp_ = plot()
    push!(p,tmp_)
    return p
end
function enssol(z_dim, tsize, nPlots, varPlot, dzdt_solve, xdata, tspan, nplot,par ;tscale=0)
    sol_ = zeros(z_dim,tsize*nPlots)
    lower_ = zeros(z_dim,tsize*nPlots)
    upper_ = zeros(z_dim,tsize*nPlots)
    #prob = ODEProblem(dzdt_rhs,xdata[:,1],(args["tspan"][1],args["tspan"][2]),par[:,1])
    
    prob_func = (prob,i,repeat) -> remake(prob,u0=prob.u0 .+
                                          varPlot .* (1.0 .- rand(z_dim))
                                          .- varPlot./2)
    for i=1:nPlots
        ind_ = (i-1)*tsize+1
        if tscale == 0
            prob = ODEProblem(dzdt_solve,xdata[:,ind_],(tspan[1],tspan[2]),par[:,i])
        else
            prob = ODEProblem(dzdt_solve,xdata[:,ind_],(tspan[1],tspan[2]),[par[:,i];tscale])
        end
        ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)
        sim = Array(solve(ensemble_prob,Tsit5(),EnsembleDistributed(),trajectories=nplot, saveat=range(tspan[1],tspan[2],length=tsize)))
        sol_[:,ind_ : (ind_ + tsize-1)] = sim[:,:,1] # add tsize to the next three lines
        lower_[:,ind_ : (ind_ + tsize-1)] = minimum(sim,dims=3)
        upper_[:,ind_ : (ind_+ tsize-1)] = maximum(sim,dims=3)
    end
    return lower_, upper_, sol_
end
function plotter(nfae::NFAE,ctr,p,z_test,alpha_,train_loss,test_loss)
    # 26/02: x changed to z_test
    # x, alpha should be cpu
    ind_z = 1:nfae.model_z.z_dim
    if nfae.nPlots != 0
        id_ = repeat(Matrix{Float32}(I,nfae.test_size,nfae.test_size),inner=(1,nfae.model_x.tsize))
        beta_ = cpu(nfae.par.encoder)(alpha_)
        # z_test = cpu(nfae.state.encoder)(x_)
        if nfae.trans != nothing
            z_test = z_test .+ cpu(nfae.trans.encoder)(beta_*id_)
        end
        lower_, upper_, sol_ =  enssol(nfae.model_z.z_dim, nfae.model_x.tsize, nfae.nPlots, nfae.varPlot,
                                nfae.model_z.solve, z_test, nfae.model_x.tspan,
                                nfae.nEnsPlot,beta_;tscale=cpu(nfae.tscale))
        for i in ind_z            
            p_ = plot()
            lab_ = []
            for j in 1:nfae.nPlots
                ind_ = nfae.model_x.tsize*(j-1)+1:(nfae.model_x.tsize*j)
                plot!(ind_,[sol_[i,ind_] sol_[i,ind_]], fillrange= [lower_[i,ind_] upper_[i,ind_]],linealpha=0,
                      fillalpha = 0.3, c=:orange,legend=:right,legendfontsize=10,lab="",xtick=[],
                      bottom_margin = 5mm,
                      ylims = (1.1*minimum([z_test lower_]),1.1*maximum([z_test upper_])))
                plot!(ind_,z_test[i,ind_],
                      title=latexstring("\\textrm{Test data: } z_$(i)"),lab="",linewidth=2,titlefont=font(15),size=(900,450),
                      c=:deepskyblue3,xtickfont = font(15),ytickfont = font(10))
                plot!(ind_,upper_[i,ind_],c=:black,lab="",linealpha=0.3)
                plot!(ind_,lower_[i,ind_],c=:black,lab="",linealpha=0.3)
                vline!(Array(ind_)[end:end],line = (:black, 0.3),lab = "")
                par_enc = beta_[1,j]
                par_ = alpha_[1,j]
                #         if sign(par_) == sign(par_enc)
                #             col_ = :green
                #         else
                #             col_ = :red
                #         end
                if sign(par_)<0
                    str_ = latexstring("\\beta < 0")
                    col_ = :green
                else
                    str_ = latexstring("\\beta > 0")
                    col_ = :red
                end
                push!(lab_,str_)
                # annotate!(ind_[Int(end/2)],1.3*minimum([z_test lower_]),text(str_,
                #                                                                  col_, :center, 12))
            end
            plot!([0],fillrange=[1 1],c=:orange,fillalpha=0.3,linealpha=0,label="Simulated")
            plot!([],c=:deepskyblue3,label="Learned")
            p[i] = p_
            
        end
        i = ind_z[end]+1
    else
        i=1
    end
    push!(p[i],[ctr],[test_loss])
    i = i+1
    push!(p[i],[ctr],[train_loss])
    i = i+1
    alpha_sort = sort(cpu(alpha_),dims=2)
    beta_sort = cpu(nfae.par.encoder)(alpha_sort)
    p[i] = Plots.scatter(beta_sort[1:nfae.model_z.par_dim,:]',markershape=:rect,markersize=4,label="Enc",title="Parameter(s)",titlefont=font(10))
    Plots.scatter!(alpha_sort[1:nfae.model_x.par_dim,:]',markershape=:utriangle,markersize=4,markeralpha=0.5,label="GT",legend=:bottomright,legendfontsize=5)

    l=0
    if nfae.nPlots == 0
        l = @layout[grid(1,1) grid(1,1) grid(1,1)]
    else
        if nfae.model_z.z_dim == 1
            l = @layout [grid(1,1); grid(1,1) grid(1,1) grid(1,1)]
        elseif nfae.model_z.z_dim == 2
            l = @layout [grid(1,1); grid(1,1); grid(1,1) grid(1,1) grid(1,1)]
        else
            l = @layout [grid(1,1); grid(1,1); grid(1,1); grid(1,1) grid(1,1) grid(1,1)]
        end
        if nfae.ijulia
            IJulia.clear_output(true) # prevents flickering
            plot(p...,layout=l) |> IJulia.display
            sleep(0.0001)
        #else
            #display(plot(p...,layout=l))
        end
    end    
return plot(p...,layout=l)    
end
