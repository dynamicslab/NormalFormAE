function train(args::Dict,train_data::Dict, test_data,NN::Dict,trained_NN::Tuple,dzdt_rhs,dzdt_solve,dzdt_sens_rhs)

    #---------------------------------------------------------------------------
    # Extract data from dictionaries
    #---------------------------------------------------------------------------
    
    x_train = train_data["x"]
    dx_train = train_data["dxdt"]
    alpha_train = train_data["alpha"]
    dxda_train = train_data["dxda"]
    dtdxda_train = train_data["dtdxda"]
    encoder = NN["encoder"]
    decoder = NN["decoder"]
    par_encoder = NN["par_encoder"]
    par_decoder = NN["par_decoder"]
    # u0_train = NN["u0_train"]
    x_test= reshape(test_data["x"],args["x_dim"],args["test_size"]*args["tsize"])
    dx_test= reshape(test_data["dx"],args["x_dim"],args["test_size"]*args["tsize"]) 
    alpha_test= test_data["alpha"]
    #mean_par = NN["mean_par"]
    
    #---------------------------------------------------------------------------
    # Printing and plotting
    #---------------------------------------------------------------------------

    function pp(x)
        return @sprintf("%.3e",x)
    end

    function gen_plot()
        p = []
        ind_ = 1:args["z_dim"]
        if args["nPlots"] != 0
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
    function enssol(nplot,xdata,par;tscale=0)
        sol_ = zeros(args["z_dim"],args["tsize"]*args["nPlots"])
        lower_ = zeros(args["z_dim"],args["tsize"]*args["nPlots"])
        upper_ = zeros(args["z_dim"],args["tsize"]*args["nPlots"])
        #prob = ODEProblem(dzdt_rhs,xdata[:,1],(args["tspan"][1],args["tspan"][2]),par[:,1])
        prob_func = (prob,i,repeat) -> remake(prob,u0=prob.u0 .+ args["varPlot"]*rand(args["z_dim"]))
        for i=1:args["nPlots"]
            ind_ = (i-1)*args["tsize"]+1
            if tscale == 0
                prob = ODEProblem(dzdt_solve,xdata[:,ind_],(args["tspan"][1],args["tspan"][2]),par[:,i])
            else
                prob = ODEProblem(dzdt_solve,xdata[:,ind_],(args["tspan"][1],args["tspan"][2]),[par[:,i];tscale])
            end
            ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)
            sim = Array(solve(ensemble_prob,Tsit5(),EnsembleDistributed(),trajectories=nplot, saveat=range(args["tspan"][1],args["tspan"][2],length=args["tsize"])))
            sol_[:,ind_ : (ind_ + args["tsize"]-1)] = sim[:,:,1] # add tsize to the next three lines
            lower_[:,ind_ : (ind_ + args["tsize"]-1)] = minimum(sim,dims=3)
            upper_[:,ind_ : (ind_+ args["tsize"]-1)] = maximum(sim,dims=3)
        end
        return lower_, upper_, sol_
    end
    function plotter(p,test_loss,train_loss,ctr,encoder,par_encoder;tscale=0)
        ind_ = 1:args["z_dim"]
        if args["nPlots"] != 0
            lower_, upper_, sol_ = enssol(args["nEnsPlot"],cpu(encoder(gpu(x_test))),cpu(par_encoder(gpu(alpha_test))),tscale=tscale)
            for i in ind_
                p[i] = plot(1:(args["tsize"]*args["nPlots"]),encoder(gpu(x_test))[i,1:args["tsize"]*args["nPlots"]],label="",title=latexstring("\\textrm{Test data: } z_$(i)"),titlefont=font(10),lab="Enc.")
                plot!([sol_[i,:] sol_[i,:]], fillrange= [lower_[i,:] upper_[i,:]], linealpha = 0.0, fillalpha = 0.3, c=:black,label="Sim.",legend=:topright,legendfontsize=5)
            end
            i = ind_[end]+1
        end
        i = 1
        push!(p[i],[ctr],[test_loss])
        i = i+1
        push!(p[i],[ctr],[train_loss])
        i = i+1
        alpha_test_ = sort(alpha_test,dims=2)
        p[i] = scatter(par_encoder(gpu(alpha_test_))',markershape=:rect,markersize=4,label="Enc",title="Parameter(s)",titlefont=font(10))
        scatter!(alpha_test_',markershape=:utriangle,markersize=4,markeralpha=0.5,label="GT",legend=:bottomright,legendfontsize=5)
    end
    ctr = 1 
    ind = Int(ceil(sqrt(args["z_dim"])))
    ind2 = ind
    if ind*ind2-args["z_dim"]<2
        ind2 = ind2+1
    end
    plot_ = gen_plot()

    #------------------------------------------------------------------------
    # Pre training
    #------------------------------------------------------------------------
    
    # Build loss
    loss_ = build_loss(args,dzdt_rhs,dzdt_solve,dzdt_sens_rhs)

    if args["pre_train"]
        # Orientation
        println("Orienting the parameters...")
        loss_orient = () -> Flux.mse(NN["par_encoder"](gpu(alpha_train))[1:args["par_dim"],:] , gpu(alpha_train)) + Flux.mse(gpu(alpha_train) ,NN["par_decoder"](NN["par_encoder"](gpu(alpha_train))))
        sign_ =  sum(abs2,sign.(NN["par_encoder"](gpu(alpha_train))) .- sign.(gpu(alpha_train)))
        while sign_>0
            ps = Flux.params(NN["par_encoder"],NN["par_decoder"])
            loss, back = Flux.pullback(ps) do
                loss_orient()
            end
        grad = back(1f0)
            Flux.Optimise.update!(ADAM(0.001),ps,grad)
            sign_ = sum(abs2,sign.(NN["par_encoder"](gpu(alpha_train))) .- sign.(gpu(alpha_train)))
            println("Sign: ",sign_)
        end
        println("Parameter orientation complete...")
    end
        
        #----------------------------------------------------------------------------------
        # Training
        #----------------------------------------------------------------------------------
        
    for i=1:args["nEpochs"]
        x_batch = 0
        dx_batch = 0
        alpha_batch = 0
        dxda_batch = 0
        dtdxda_batch = 0
        println("Epoch ",i)
        nbatches = div(args["training_size"],args["BatchSize"])
        for j=1:nbatches
            
            # Extract batch data
            encoder = NN["encoder"]
            decoder = NN["decoder"]
            par_encoder = NN["par_encoder"]
            par_decoder = NN["par_decoder"]
            println("L_Batch ",j)
            batchsize = args["BatchSize"]
            index1 = (j-1)*batchsize+1
            index2 = j*batchsize
            ind_ = shuffle(index1:index2)
            #ind_ = index1:index2
            x_batch = hcat([x_train[:,:,i] for i in ind_]...) |> gpu
            dx_batch = hcat([dx_train[:,:,i] for i in ind_]...) |> gpu
            alpha_batch = hcat([alpha_train[:,i] for i in ind_]...) |> gpu
            # dxda_batch = dxda_train[:,:,:,ind_] |> gpu
            # dtdxda_batch = dtdxda_train[:,:,:,ind_] |> gpu
            dxda_batch = rand(1)
            dtdxda_batch = rand(1)
            loss = 0.0f0

            # Compute test loss
            enc_ = NN["encoder"](gpu(test_data["x"]))
            NN["u0_train"] = hcat([enc_[:,(i-1)*args["tsize"]+1] for i in 1:args["test_size"]]...) |> gpu
            test_loss = loss_(NN["encoder"],NN["decoder"],NN["par_encoder"],NN["par_decoder"],NN["u0_train"],NN["tscale"],gpu(test_data["x"]),gpu(test_data["dx"]),gpu(test_data["alpha"]),rand(1),rand(1),1)
            println("  Test loss: $(pp(args["loss_total"])) AE:  $(pp(args["loss_AE"])) dxdt: $(pp(args["loss_dxdt"])) dzdt: $(pp(args["loss_dzdt"])) par: $(pp(args["loss_par"])) NLRAN_in: $(pp(args["loss_NLRAN_in"])) NLRAN_out: $(pp(args["loss_NLRAN_out"])) Orientation: $(pp(args["loss_orient"])) Zero function: $(pp(args["loss_zero"]))")
            println("  Relative test loss: $(pp(args["rel_loss_total"])) AE:  $(pp(args["rel_loss_AE"])) dxdt: $(pp(args["rel_loss_dxdt"])) dzdt: $(pp(args["rel_loss_dzdt"])) par: $(pp(args["rel_loss_par"])) NLRAN_in: $(pp(args["rel_loss_NLRAN_in"])) NLRAN_out: $(pp(args["rel_loss_NLRAN_out"])) Orientation: $(pp(args["rel_loss_orient"])) Zero function: $(pp(args["rel_loss_zero"]))")
            
            # Training
            enc_ = NN["encoder"](x_batch)
            NN["u0_train"] = hcat([enc_[:,(i-1)*args["tsize"]+1] for i in 1:args["BatchSize"]]...) |> gpu
            for i=1:args["nIt"]
                #Flux.train!(loss_,Flux.params(encoder,decoder,hom_encoder,hom_decoder),data_,ADAM(args["ADAMarg"]))
                ps = Flux.params(trained_NN...)
                loss, back = Flux.pullback(ps) do
                     loss_(NN["encoder"],NN["decoder"],NN["par_encoder"],NN["par_decoder"],NN["u0_train"],NN["tscale"],x_batch,dx_batch,alpha_batch,dxda_batch,dtdxda_batch,0)
                end
                grad = back(1f0)
                Flux.Optimise.update!(ADAM(args["ADAMarg"]),ps,grad)
            end
            # train tscale
            enc_ = NN["encoder"](x_batch)
            NN["u0_train"] = hcat([enc_[:,(i-1)*args["tsize"]+1] for i in 1:args["BatchSize"]]...) |> gpu
            for i=1:args["nIt_tscale"]
                ps = Flux.params(NN["tscale"])
                loss, back = Flux.pullback(ps) do
                     loss_(NN["encoder"],NN["decoder"],NN["par_encoder"],NN["par_decoder"],NN["u0_train"],NN["tscale"],x_batch,dx_batch,alpha_batch,dxda_batch,dtdxda_batch,0)
                end
                grad = back(1f0)
                Flux.Optimise.update!(ADAM(args["ADAMarg"]),ps,grad)
            end

            # Display Training loss
            println("  Train loss: $(pp(args["loss_total"])) AE:  $(pp(args["loss_AE"])) dxdt: $(pp(args["loss_dxdt"])) dzdt: $(pp(args["loss_dzdt"])) par: $(pp(args["loss_par"])) NLRAN_in: $(pp(args["loss_NLRAN_in"])) NLRAN_out: $(pp(args["loss_NLRAN_out"])) Orientation: $(pp(args["loss_orient"])) Zero function: $(pp(args["loss_zero"]))")
            println("  Relative train loss: $(pp(args["rel_loss_total"])) AE:  $(pp(args["rel_loss_AE"])) dxdt: $(pp(args["rel_loss_dxdt"])) dzdt: $(pp(args["rel_loss_dzdt"])) par: $(pp(args["rel_loss_par"])) NLRAN_in: $(pp(args["rel_loss_NLRAN_in"])) NLRAN_out: $(pp(args["rel_loss_NLRAN_out"])) Orientation: $(pp(args["rel_loss_orient"])) Zero function: $(pp(args["rel_loss_zero"]))")

            # Deploy plotting
            plotter(plot_,test_loss,loss,ctr,NN["encoder"],NN["par_encoder"],tscale=cpu(NN["tscale"]))
            l=0
            if args["nPlots"] == 0
                l = @layout[grid(1,1) grid(1,1) grid(1,1)]
            else
                if args["z_dim"] == 1
                    l = @layout [grid(1,1); grid(1,1) grid(1,1) grid(1,1)]
                elseif args["z_dim"] == 2
                    l = @layout [grid(1,1); grid(1,1); grid(1,1) grid(1,1) grid(1,1)]
                else
                    l = @layout [grid(1,1); grid(1,1); grid(1,1); grid(1,1) grid(1,1) grid(1,1)]
                end
                display(plot(plot_...,layout=l))
                ctr = ctr+1
            end
        end       
        
    end
end

            
    
