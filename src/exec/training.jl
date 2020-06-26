function train(args::Dict,train_data::Dict, test_data::Dict, NN::Dict,trained_NN1::Tuple,trained_NN2::Tuple,rhs,rhs_solve)
    x_train = train_data["x"]
    dx_train = train_data["dx"]
    alpha_train = train_data["alpha"]
    x_test = test_data["x"]
    dx_test = test_data["dx"]
    alpha_test = test_data["alpha"]
    encoder = NN["encoder"]
    decoder = NN["decoder"]
    par_encoder = NN["par_encoder"]
    par_decoder = NN["par_decoder"]
    u0_train = NN["u0_train"]
    function gen_plot()
        p = []
        ind_ = 1:args["z_dim"]
        for i in ind_
            tmp_ = plot()
            push!(p,tmp_)
        end
        tmp_ = plot([0.0],[0.0],marker=2,title="Test loss")
        push!(p,tmp_)
        tmp_ = plot([0.0],[0.0],marker=2,title="Train loss")
        push!(p,tmp_)
        tmp_ = plot()
        push!(p,tmp_)
        return p
    end
    function plotter(p,test_loss,train_loss,ctr)
        ind_ = 1:args["z_dim"]
        for i in ind_
            p[i] = plot(1:500,cpu(encoder(x_test)[i,1:500]),label="z$(i) tr.",title="Test data")
            plot!(p[i],1:500,cpu(test_data["z"][i,1:500]),label="z$(i) GT")
        end
        i = ind_[end]+1
        push!(p[i],[ctr],[test_loss])
        i = i+1
        push!(p[i],[ctr],[train_loss])
        i = i+1
        len_ = args["tsize"]*args["test_size"]
        p[i] = plot(1:len_,par_encoder(test_data["alpha"])[1:len_],label="Encoded")
        plot!(1:len_,test_data["par"][1:len_],label="GT")
    end
    ctr = 1 
    ind = Int(ceil(sqrt(args["z_dim"])))
    ind2 = ind
    if ind*ind2-args["z_dim"]<2
        ind2 = ind2+1
    end
    plot_ = gen_plot()
    for i=1:args["nEpochs"]
        x_batch = 0
        dx_batch = 0
        alpha_batch = 0
        println("Epoch ",i)
        #for j=1:args["nBatches"]
        for j=1:args["training_size"]
            println("L_Batch ",j)
            #batchsize = div(args["training_size"],args["nBatches"])
            #index1 = (j-1)*batchsize+1
            #index2 = j*batchsize
            #ind_ = shuffle(index1:index2)
            #ind_ = index1:index2
            x_batch = x_train[j,:,:] |> gpu
            #for val in ind_[2:end]
            #    x_batch = [x_batch x_train[val,:,:]] |> gpu
            #end
            dx_batch = dx_train[j,:,:] |> gpu
            #for val in ind_[2:end]
            #    dx_batch = [dx_batch dx_train[val,:,:]] |> gpu
            #end
            alpha_batch = alpha_train[j,:,:] |> gpu
            #for val in ind_[2:end]
            #    alpha_batch = [alpha_batch alpha_train[val,:,:]] |> gpu
            #end
            loss_ = build_loss(args,rhs,rhs_solve,encoder,decoder,par_encoder,par_decoder,u0_train)
            loss = 0.0f0
            # args["batchsize"] = args["test_size"]
            test_loss = 0.0f0
            for k in 1:args["test_size"]
                i1 = (k-1)*args["tsize"]
                i2 = k*args["tsize"]
                test_loss += loss_(x_test[:,i1+1:i2],dx_test[:,i1+1:i2],alpha_test[:,i1+1:i2])
            end
            println("  Test loss: ",test_loss," AE: ",args["loss_AE"]," dxdt: ",args["loss_dxdt"], " dzdt: ",args["loss_dzdt"], " par: ",args["loss_par"])
            data_ = [(x_batch,dx_batch,alpha_batch)] 
            # args["batchsize"] = div(args["training_size"],args["nBatches"])
            # Train 1
            for i=1:args["nIt_AE"]
                #Flux.train!(loss_,Flux.params(encoder,decoder,hom_encoder,hom_decoder),data_,ADAM(args["ADAMarg"]))
                ps = Flux.params(trained_NN1...)
                loss, back = Flux.pullback(ps) do
                    loss_(x_batch,dx_batch,alpha_batch)
                end
                grad = back(1f0)
                Flux.Optimise.update!(Flux.Optimiser(ADAM(args["ADAMarg"]),WeightDecay(0.01)),ps,grad)
            end
            # Train 2
            for i=1:args["nIt_u0"]
                #Flux.train!(loss_,Flux.params(encoder,decoder,hom_encoder,hom_decoder),data_,ADAM(args["ADAMarg"]))
                ps = Flux.params(u0_train)
                loss, back = Flux.pullback(ps) do
                    loss_(x_batch,dx_batch,alpha_batch)
                end
                grad = back(1f0)
                Flux.Optimise.update!(Flux.Optimiser(ADAM(args["ADAMarg"]),WeightDecay(0.01)),ps,grad)
            end
            # Train 3
            for i=1:args["nIt_par"]
                #Flux.train!(loss_,Flux.params(encoder,decoder,hom_encoder,hom_decoder),data_,ADAM(args["ADAMarg"]))
                ps = Flux.params(trained_NN2...)
                loss, back = Flux.pullback(ps) do
                    loss_(x_batch,dx_batch,alpha_batch)
                end
                grad = back(1f0)
                Flux.Optimise.update!(Flux.Optimiser(ADAM(args["ADAMarg"]),WeightDecay(0.01)),ps,grad)
            end
            println("  Train loss: ",args["loss_total"]," AE: ",args["loss_AE"]," dxdt: ",args["loss_dxdt"], " dzdt: ",args["loss_dzdt"], " par: ",args["loss_par"])
            plotter(plot_,test_loss,loss,ctr)
            display(plot(plot_...))
            ctr = ctr+1
        end       
        
    end
end

            
    
