function train(args::Dict,train_data::Dict, test_data::Dict, NN::Dict,rhs)
    x_train = train_data["x"]
    dx_train = train_data["dx"]
    x_test = test_data["x"]
    dx_test = test_data["dx"]
    encoder = NN["encoder"]
    decoder = NN["decoder"]
    hom_encoder = NN["hom_encoder"]
    hom_decoder = NN["hom_decoder"]
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
            loss_ = build_loss(args,rhs,encoder,decoder)
            test_loss=loss_(x_test,dx_test)
            println("  Test loss: ",test_loss," AE: ",args["loss_AE"]," dxdt: ",args["loss_dxdt"], " dzdt: ",args["loss_dzdt"])
            data_ = [(x_batch,dx_batch)]
            for i=1:args["nIterations"]
                #Flux.train!(loss_,Flux.params(encoder,decoder,hom_encoder,hom_decoder),data_,ADAM(args["ADAMarg"]))
                ps = Flux.params(encoder,decoder)
                loss, back = Flux.pullback(ps) do
                    loss_(x_batch,dx_batch)
                end
                grad = back(1f0)
                Flux.Optimise.update!(ADAM(0.001),ps,grad)
                println("  Train loss: ",args["loss_total"]," AE: ",args["loss_AE"]," dxdt: ",args["loss_dxdt"], " dzdt: ",args["loss_dzdt"])
            end
        end
    end
    NN["encoder"] = encoder
    NN["decoder"] = decoder
    NN["hom_encoder"] = hom_encoder
    NN["hom_decoder"] = hom_decoder
end

            
    
