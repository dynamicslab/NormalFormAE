rel_loss(nfae_, x) = Flux.mse(nfae_.state.decoder(nfae_.state.encoder(x))./x,x./x)

function train(nfae,nEpochs, batchsize,x_test,dx_test,alpha_test)
    global ctr,opt,adamarg,last_loss,loss_test,rel_loss_test, loss_train_full, loss_test_full
    loss_train_new = 0
    for i in 1:nEpochs
        for j in 1:div(training_size, batchsize)
            x_, dx_, alpha_ = makebatch(nfae.training_data, batchsize,j) |> nfae.machine # batcher
            println(size(x_))
            ps = Flux.params(nfae.state.encoder, nfae.state.decoder,
                            nfae.par.encoder, nfae.par.decoder,nfae.tscale) # defines NNs to be trained
            res, back = Flux.pullback(ps) do
                nfae(x_,dx_,alpha_) 
            end # Performs autodiff step
            Flux.Optimise.update!(opt,ps,back(1f0)) # Updates training parameters
            loss_train = res
            push!(loss_train_full,nfae.loss)
            loss_test = nfae(x_test,dx_test,alpha_test)
            push!(loss_test_full,nfae.loss)
            z_ = nfae.state.encoder(x_test)
            @printf "Epoch: %i, Batch: %i, eta: %0.1e, lowest rel loss: %0.3e \n" i j adamarg rel_loss_test  
            @printf "Train loss: %0.3e, %0.3e, %0.3e, %0.3e, %0.3e %0.3e \n" loss_train_full[ctr]... 
            @printf "Test loss: %0.3e, %0.3e, %0.3e, %0.3e, %0.3e %0.3e \n" loss_test_full[ctr]... 
            try
                #plotter(nfae,ctr,p,cpu(z_),cpu(alpha_test),loss_train,loss_test)
                #savefig("NeuralFieldTrain.pdf")
            catch e
            end
            pars_off = (sum(abs.(sign.(nfae.par.encoder(alpha_test)) .- sign.(alpha_test))))/2
            println("Pars off: $(pars_off)")
            ctr = ctr + 1
            try
                files_ = readdir("/tmp/")
                for elem in files_
                    if occursin("jl_",elem) & occursin(".svg",elem)
                        rm("/tmp/$(elem)")
                    end
                end
            catch e
            end
        end
        pars_off = (sum(abs.(sign.(nfae.par.encoder(alpha_test)) .- sign.(alpha_test))))/2
        if pars_off > 5.0f0 # <5 only for transcritical
            flag = 1
            #break
        end
        #if loss_test < last_loss && rel_loss(nfae,x_test) < rel_loss_test
        #    save_params(nfae)
        #    last_loss = loss_test
        #    rel_loss_test = rel_loss(nfae,x_test)
        #else
        #    if i > 10
        #    adamarg = 0.1*adamarg
        #    opt = ADAM(adamarg)
        #    end
        #end
        FileIO.save("test_loss.jld2","test_loss",loss_test_full)
        FileIO.save("train_loss.jld2","train_loss",loss_train_full)
        #if rel_loss_test < 5e-2 || adamarg < 1e-6
        #    break
        #end    
    end
    nothing
end
