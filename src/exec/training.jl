function train(args::Dict,train_data::Dict, test_data::Dict, NN::Dict,trained_NN::Tuple,rhs)
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
        for j=1:args["nBatches"]
            println("L_Batch ",j)
            batchsize = div(args["training_size"],args["nBatches"])
            index1 = (j-1)*batchsize+1
            index2 = j*batchsize
            #ind_ = shuffle(index1:index2)
            ind_ = index1:index2
            x_batch = x_train[ind_[1],:,:]
            for val in ind_[2:end]
                x_batch = [x_batch x_train[val,:,:]] |> gpu
            end
            dx_batch = dx_train[ind_[1],:,:]
            for val in ind_[2:end]
                dx_batch = [dx_batch dx_train[val,:,:]] |> gpu
            end
            alpha_batch = alpha_train[ind_[1],:,:]
            for val in ind_[2:end]
                alpha_batch = [alpha_batch alpha_train[val,:,:]] |> gpu
            end
            loss_ = build_loss(args,rhs,encoder,decoder,par_encoder,par_decoder)
            loss = 0.0f0
            test_loss=loss_(x_test,dx_test,alpha_test)
            println("  Test loss: ",test_loss," AE: ",args["loss_AE"]," dxdt: ",args["loss_dxdt"], " dzdt: ",args["loss_dzdt"], " par: ",args["loss_par"])
            data_ = [(x_batch,dx_batch,alpha_batch)]
            for i=1:args["nIterations"]
                #Flux.train!(loss_,Flux.params(encoder,decoder,hom_encoder,hom_decoder),data_,ADAM(args["ADAMarg"]))
                ps = Flux.params(trained_NN...)
                loss, back = Flux.pullback(ps) do
                    loss_(x_batch,dx_batch,alpha_batch)
                end
                grad = back(1f0)
                Flux.Optimise.update!(ADAM(args["ADAMarg"]),ps,grad)
            end
            println("  Train loss: ",args["loss_total"]," AE: ",args["loss_AE"]," dxdt: ",args["loss_dxdt"], " dzdt: ",args["loss_dzdt"], " par: ",args["loss_par"])
            plotter(plot_,test_loss,loss,ctr)
            display(plot(plot_...))
            ctr = ctr+1
        end
        # Refinement
        # for i=1:args["nIterations"]
        #     #Flux.train!(loss_,Flux.params(encoder,decoder,hom_encoder,hom_decoder),data_,ADAM(args["ADAMarg"]))
        #     ps = Flux.params(lin_trans)
        #     loss, back = Flux.pullback(ps) do
        #         loss_(x_batch,dx_batch,alpha_batch)
        #     end
        #     grad = back(1f0)
        #     Flux.Optimise.update!(ADAM(args["ADAMarg"]),ps,grad)
        # end
        
        
    end
end

            
    
