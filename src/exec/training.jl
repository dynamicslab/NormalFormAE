function train(args::Dict,train_data::Dict, test_data::Dict, NN::Dict,rhs)
    x_train = train_data["x"]
    dx_train = train_data["dx"]
    par_train = train_data["par"]
    x_test = test_data["x"]
    dx_test = test_data["dx"]
    par_test = test_data["par"]
    encoder = NN["encoder"]
    decoder = NN["decoder"]
    par_encoder = NN["par_encoder"]
    par_decoder = NN["par_decoder"]
    for i=1:args["nEpochs"]
        println("Epoch ",i)
        for j=1:args["nBatches"]
            println("L_Batch ",j)
            batchsize = div(args["training_size"],args["nBatches"])
            index1 = (j-1)*batchsize+1
            index2 = j*batchsize
            ind_ = shuffle(index1:index2)
            x_batch = x_train[ind_[1],:,:]
            for val in ind_[2:end]
                x_batch = [x_batch x_train[val,:,:]] |> gpu
            end
            dx_batch = dx_train[ind_[1],:,:]
            for val in ind_[2:end]
                dx_batch = [dx_batch dx_train[val,:,:]] |> gpu
            end
            par_batch = par_train[ind_[1],:,:]
            for val in ind_[2:end]
                par_batch = [par_batch par_train[val,:,:]] |> gpu
            end
            loss_ = build_loss(args,rhs,encoder,decoder,par_encoder,par_decoder)
            test_loss=loss_(x_test,dx_test,par_test)
            println("  Test loss: ",test_loss," AE: ",args["loss_AE"]," dxdt: ",args["loss_dxdt"], " dzdt: ",args["loss_dzdt"], " par: ",args["loss_par"])
            data_ = [(x_batch,dx_batch,par_batch)]
            for i=1:args["nIterations"]
                #Flux.train!(loss_,Flux.params(encoder,decoder,hom_encoder,hom_decoder),data_,ADAM(args["ADAMarg"]))
                ps = Flux.params(encoder,decoder)
                loss, back = Flux.pullback(ps) do
                    loss_(x_batch,dx_batch,par_batch)
                end
                grad = back(1f0)
                Flux.Optimise.update!(ADAM(args["ADAMarg"]),ps,grad)
            end
            println("  Train loss: ",args["loss_total"]," AE: ",args["loss_AE"]," dxdt: ",args["loss_dxdt"], " dzdt: ",args["loss_dzdt"], " par: ",args["loss_par"])
        end
    end
end

            
    
