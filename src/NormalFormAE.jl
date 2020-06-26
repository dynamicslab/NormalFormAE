module NormalFormAE

using DifferentialEquations, Flux, Distributions, Plots, CUDAapi, Random, PolyChaos, Zygote, DiffEqFlux


include("exec/autoencoder.jl")
include("exec/training.jl")
include("exec/data_gen.jl")

export gen

export get_autoencoder, dt_NN, build_loss

export train, pre_train

function pre_train(args::Dict,rhs)
    training_data = Dict()
    test_data = Dict()
    losses_ = Dict()
    NN = Dict()
    
    encoder, decoder, par_encoder, par_decoder, u0_train = get_autoencoder(args) 
    t_train,z_train,dz_train,x_train,dx_train,par_train,alpha_train = gen(args,rhs,args["training_size"])
    t_test,z_test,dz_test,x_test,dx_test,par_test,alpha_test = gen(args,rhs,args["test_size"],"test")

    NN["encoder"] = encoder |> gpu
    NN["decoder"] = decoder |> gpu
    NN["par_decoder"] = par_decoder |> gpu 
    NN["par_encoder"] = par_encoder |> gpu
    NN["u0_train"] = u0_train |> gpu
    
    training_data["t"] = t_train
    training_data["z"] = z_train 
    training_data["dz"] = dz_train
    training_data["x"] = x_train
    training_data["dx"] = dx_train
    training_data["par"] = par_train
    training_data["alpha"] = alpha_train
    
    test_data["t"] = t_test |> gpu
    test_data["z"] = z_test |> gpu
    test_data["dz"] = dz_test |> gpu
    test_data["x"] = x_test |> gpu
    test_data["dx"] = dx_test |> gpu
    test_data["par"] = par_test |> gpu
    test_data["alpha"] = alpha_test |> gpu
    
    return NN, training_data, test_data
end

end
