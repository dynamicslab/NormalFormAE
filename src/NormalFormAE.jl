module NormalFormAE

using DifferentialEquations, Flux, Distributions, Plots, CUDAapi, Random, PolyChaos, Zygote, DiffEqFlux, DiffEqSensitivity, LinearAlgebra


include("exec/autoencoder.jl")
include("exec/training.jl")
include("exec/data_gen.jl")

export gen

export get_autoencoder, dt_NN, build_loss

export train, pre_train

function pre_train(args::Dict,rhs,sens_rhs)
    training_data = Dict()
    test_data = Dict()
    losses_ = Dict()
    NN = Dict()
    
    encoder, decoder, par_encoder, par_decoder, u0_train = get_autoencoder(args) 
    x_train,dx_train,alpha_train, dxda_train, dtdxda_train = gen(args,rhs,sens_rhs,args["training_size"])
    x_test,dx_test,alpha_test, dxda_test, dtdxda_test = gen(args,rhs,sens_rhs,args["test_size"])

    NN["encoder"] = encoder |> gpu
    NN["decoder"] = decoder |> gpu
    NN["par_decoder"] = par_decoder |> gpu 
    NN["par_encoder"] = par_encoder |> gpu
    NN["u0_train"] = u0_train |> gpu
    #NN["mean_par"] = Float32.(rand(args["par_dim"])) |> gpu
    
    training_data["x"] = x_train
    training_data["dxdt"] = dx_train
    training_data["alpha"] = alpha_train
    training_data["dxda"] = dxda_train
    training_data["dtdxda"] = dtdxda_train

    test_data["x"] = hcat([x_test[:,:,i] for i in 1:args["test_size"]]...) |> gpu
    test_data["dx"] =  hcat([dx_test[:,:,i] for i in 1:args["test_size"]]...) |> gpu
    test_data["alpha"] = hcat([alpha_test[:,i] for i in 1:args["test_size"]]...) |> gpu
    
    return NN, training_data, test_data
end

end
