# using Pkg
# Pkg.activate(".")
# using Flux
# using Random

# include("../models/LP.jl")
# include("autoencoder.jl")
# include("data_gen.jl")

# params = Dict()
# params["P_DataFid"] = 1.0
# params["P_Hom"] = 1.0
# params["P_dx"] = 1.0
# params["P_dz"] = 1.0
# params["AE_widths"] = [128,64,32,16,2]
# params["AE_acts"] = ["sigmoid","sigmoid","sigmoid","id"]
# params["Hom_widths"] = [2,2,2]
# params["Hom_acts"] = ["sigmoid","id"]
# params["z_dim"] = 2
# params["expansion_order"] = 1
# params["spatial_scale"] = 128
# params["tspan"] =  [0.0,0.1]
# params["tsize"] = 100
# params["mean_init"] = [0.01,0.0]

# params["training_size"] = 1000
# params["test_size"] = 20
# params["nEpochs"] = 10
# params["nBatches"] = 5
# params["nIterations"] = 100
# params["ADAMarg"] = 0.001

# training_data = Dict()
# test_data = Dict()
# losses_ = Dict()
# NN = Dict()

# encoder, decoder, hom_encoder, hom_decoder = get_autoencoder(params)
# t_train,z_train,dz_train,x_train,dx_train = gen(params,rhs,params["training_size"],2)
# t_test,z_test,dz_test,x_test,dx_test = gen(params,rhs,params["test_size"],2,"test")



# NN["encoder"] = encoder
# NN["decoder"] = decoder
# NN["hom_decoder"] = hom_decoder
# NN["hom_encoder"] = hom_encoder

# training_data["t"] = t_train
# training_data["z"] = z_train
# training_data["dz"] = dz_train
# training_data["x"] = x_train
# training_data["dx"] = dx_train

# test_data["t"] = t_test
# test_data["z"] = z_test
# test_data["dz"] = dz_test
# test_data["x"] = x_test
# test_data["dx"] = dx_test


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
                x_batch = [x_batch x_train[val,:,:]]
            end
            dx_batch = dx_train[ind_[1],:,:]
            for val in ind_[2:end]
                dx_batch = [dx_batch dx_train[val,:,:]]
            end
            loss_ = build_loss(args,rhs,encoder,decoder,hom_encoder,hom_decoder)
            test_loss=loss_(x_test,dx_test)
            println("  Test loss: ",test_loss," AE: ",args["loss_AE"]," Hom: ",args["loss_Hom"]," dxdt: ",args["loss_dxdt"], " dzdt: ",args["loss_dzdt"])
            data_ = [(x_batch,dx_batch)]
            for i=1:args["nIterations"]
                Flux.train!(loss_,Flux.params(encoder,decoder,hom_encoder,hom_decoder),data_,ADAM(args["ADAMarg"]))
                println("  Train loss: ",args["loss_total"]," AE: ",args["loss_AE"]," Hom: ",args["loss_Hom"]," dxdt: ",args["loss_dxdt"], " dzdt: ",args["loss_dzdt"])
            end
        end
    end
    NN["encoder"] = encoder
    NN["decoder"] = decoder
    NN["hom_encoder"] = hom_encoder
    NN["hom_decoder"] = hom_decoder
end

            
    
