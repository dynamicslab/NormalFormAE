function train(args::Dict,train_data::Dict, test_data,NN::Dict,trained_NN1::Tuple,trained_NN2::Tuple,dzdt_rhs,dzdt_sens_rhs)
    x_train = train_data["x"]
    dx_train = train_data["dxdt"]
    alpha_train = train_data["alpha"]
    dxda_train = train_data["dxda"]
    dtdxda_train = train_data["dtdxda"]
    encoder = NN["encoder"]
    decoder = NN["decoder"]
    par_encoder = NN["par_encoder"]
    par_decoder = NN["par_decoder"]
    u0_train = NN["u0_train"]
    x_test= reshape(test_data["x"],args["x_dim"],args["test_size"]*args["tsize"])
    dx_test= reshape(test_data["dx"],args["x_dim"],args["test_size"]*args["tsize"]) 
    alpha_test= test_data["alpha"]
    #mean_par = NN["mean_par"]

    
    function gen_plot()
        p = []
        ind_ = 1:args["z_dim"]
        for i in ind_
            tmp_ = plot()
            push!(p,tmp_)
        end
        tmp_ = plot([0.0],[0.0],marker=2,title="Test loss",titlefont=font(10),label="")
        push!(p,tmp_)
        tmp_ = plot([0.0],[0.0],marker=2,title="Train loss",titlefont=font(10),label="")
        push!(p,tmp_)
        tmp_ = plot()
        push!(p,tmp_)
        return p
    end
    function enssol(nplot,xdata,par)
        sol_ = zeros(args["z_dim"],args["tsize"]*args["nPlots"])
        lower_ = zeros(args["z_dim"],args["tsize"]*args["nPlots"])
        upper_ = zeros(args["z_dim"],args["tsize"]*args["nPlots"])
        prob = ODEProblem(dzdt_rhs,xdata[:,1],(args["tspan"][1],args["tspan"][2]),par[:,1])
        prob_func = (prob,i,repeat) -> remake(prob,u0=prob.u0 .+ args["varPlot"]*rand(args["z_dim"]))
        for i=1:args["nPlots"]
            ind_ = (i-1)*args["tsize"]+1
            prob = ODEProblem(dzdt_rhs,xdata[:,ind_],(args["tspan"][1],args["tspan"][2]),par[:,i])
            ensemble_prob = EnsembleProblem(prob,prob_func=prob_func)
            sim = Array(solve(ensemble_prob,Tsit5(),EnsembleDistributed(),trajectories=nplot, saveat=range(args["tspan"][1],args["tspan"][2],length=args["tsize"])))
            sol_[:,ind_ : (ind_ + args["tsize"]-1)] = sim[:,:,1] # add tsize to the next three lines
            lower_[:,ind_ : (ind_ + args["tsize"]-1)] = minimum(sim,dims=3)
            upper_[:,ind_ : (ind_+ args["tsize"]-1)] = maximum(sim,dims=3)
        end
        return lower_, upper_, sol_
    end
    function plotter(p,test_loss,train_loss,ctr,encoder,par_encoder)
        ind_ = 1:args["z_dim"]
        lower_, upper_, sol_ = enssol(args["nEnsPlot"],cpu(encoder(gpu(x_test))),cpu(par_encoder(gpu(alpha_test))))
        for i in ind_
            p[i] = plot(1:(args["tsize"]*args["nPlots"]),encoder(gpu(x_test))[i,1:args["tsize"]*args["nPlots"]],label="",title="Test data",titlefont=font(10))
            plot!([sol_[i,:] sol_[i,:]], fillrange= [lower_[i,:] upper_[i,:]], linealpha = 0.0, fillalpha = 0.3, c=:gray,label="")
        end
        i = ind_[end]+1
        push!(p[i],[ctr],[test_loss])
        i = i+1
        push!(p[i],[ctr],[train_loss])
        i = i+1
        #len_ = args["tsize"]*args["nPlots"]
        #id_ = repeat(Matrix(I,args["nPlots"],args["nPlots"]),inner=(1,args["tsize"]))
        #p[i] = plot(1:len_,((par_encoder(gpu(alpha_test))[:,1:args["nPlots"]])*id_)',label="Encoded")
        #plot!(1:len_,(alpha_test[:,1:args["nPlots"]]*id_)',label="GT")
        alpha_test_ = sort(alpha_test,dims=2)
        p[i] = scatter(par_encoder(gpu(alpha_test_))',markershape=:rect,markersize=4,label="")
        scatter!(alpha_test_',markershape=:utriangle,markersize=8,markeralpha=0.5,label="GT",legend=:bottomright,legendfontsize=5)
    end
    ctr = 1 
    ind = Int(ceil(sqrt(args["z_dim"])))
    ind2 = ind
    if ind*ind2-args["z_dim"]<2
        ind2 = ind2+1
    end
    plot_ = gen_plot()
    loss_ = build_loss(args,dzdt_rhs,dzdt_sens_rhs)
    
    for i=1:args["nEpochs"]
        x_batch = 0
        dx_batch = 0
        alpha_batch = 0
        dxda_batch = 0
        dtdxda_batch = 0
        println("Epoch ",i)
        nbatches = div(args["training_size"],args["BatchSize"])
        for j=1:nbatches
            encoder = NN["encoder"]
            decoder = NN["decoder"]
            par_encoder = NN["par_encoder"]
            par_decoder = NN["par_decoder"]
            u0_train = NN["u0_train"]
            println("L_Batch ",j)
            batchsize = args["BatchSize"]
            index1 = (j-1)*batchsize+1
            index2 = j*batchsize
            #ind_ = shuffle(index1:index2)
            ind_ = index1:index2
            x_batch = hcat([x_train[:,:,i] for i in ind_]...) |> gpu
            dx_batch = hcat([dx_train[:,:,i] for i in ind_]...) |> gpu
            alpha_batch = hcat([alpha_train[:,i] for i in ind_]...) |> gpu
            
            # dxda_batch = dxda_train[:,:,:,ind_] |> gpu
            # dtdxda_batch = dtdxda_train[:,:,:,ind_] |> gpu
            dxda_batch = rand(1)
            dtdxda_batch = rand(1)
            loss = 0.0f0
            enc_ = NN["encoder"](gpu(test_data["x"]))
            NN["u0_train"] = hcat([enc_[:,(i-1)*args["tsize"]+1] for i in 1:args["test_size"]]...) |> gpu
            for i=1:1
                #Flux.train!(loss_,Flux.params(encoder,decoder,hom_encoder,hom_decoder),data_,ADAM(args["ADAMarg"]))
                ps = Flux.params(u0_train)
                loss, back = Flux.pullback(ps) do
                    loss_(NN["encoder"],NN["decoder"],NN["par_encoder"],NN["par_decoder"],NN["u0_train"],gpu(test_data["x"]),gpu(test_data["dx"]),gpu(test_data["alpha"]),rand(1),rand(1),testt=1)
                end
                grad = back(1f0)
                Flux.Optimise.update!(ADAM(args["ADAMarg"]),ps,grad)
            end
            test_loss = loss    
            println("  Test loss: ",args["loss_total"]," AE: ",args["loss_AE"]," dxdt: ",args["loss_dxdt"], " dzdt: ",args["loss_dzdt"], " par: ",args["loss_par"],  " sens_x: ",args["loss_sens_x"],  " sens_dt: ",args["loss_sens_dt"], " NLRAN_in: ", args["loss_NLRAN_in"], " NLRAN_out: ",args["loss_NLRAN_out"], " u0: ",args["loss_u0"])
            # args["batchsize"] = div(args["training_size"],args["nBatches"])

            
            # Train 1
            enc_ = NN["encoder"](x_batch)
            NN["u0_train"] = hcat([enc_[:,(i-1)*args["tsize"]+1] for i in 1:args["BatchSize"]]...) |> gpu
            for i=1:args["nIt_1"]
                #Flux.train!(loss_,Flux.params(encoder,decoder,hom_encoder,hom_decoder),data_,ADAM(args["ADAMarg"]))
                ps = Flux.params(trained_NN1...)
                loss, back = Flux.pullback(ps) do
                    loss_(NN["encoder"],NN["decoder"],NN["par_encoder"],NN["par_decoder"],NN["u0_train"],x_batch,dx_batch,alpha_batch,dxda_batch,dtdxda_batch)
                end
                grad = back(1f0)
                Flux.Optimise.update!(Flux.Optimiser(ADAM(args["ADAMarg"]),WeightDecay(0.01)),ps,grad)
            end
            # Train 2
            for i=1:args["nIt_2"]
                #Flux.train!(loss_,Flux.params(encoder,decoder,hom_encoder,hom_decoder),data_,ADAM(args["ADAMarg"]))
                ps = Flux.params(trained_NN2...)
                loss, back = Flux.pullback(ps) do
                     loss_(NN["encoder"],NN["decoder"],NN["par_encoder"],NN["par_decoder"],NN["u0_train"],x_batch,dx_batch,alpha_batch,dxda_batch,dtdxda_batch)
                end
                grad = back(1f0)
                Flux.Optimise.update!(Flux.Optimiser(ADAM(args["ADAMarg"]),WeightDecay(0.01)),ps,grad)
            end
            println("  Train loss: ",args["loss_total"]," AE: ",args["loss_AE"]," dxdt: ",args["loss_dxdt"], " dzdt: ",args["loss_dzdt"], " par: ",args["loss_par"],  " sens_x: ",args["loss_sens_x"],  " sens_dt: ",args["loss_sens_dt"], " NLRAN_in: ", args["loss_NLRAN_in"], " NLRAN_out: ",args["loss_NLRAN_out"], " u0: ",args["loss_u0"])
            plotter(plot_,test_loss,loss,ctr,NN["encoder"],NN["par_encoder"])
            l=0
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

            
    
