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
    
    encoder, decoder, par_encoder, par_decoder = get_autoencoder(args) 
    x_train,dx_train,alpha_train, dxda_train, dtdxda_train = gen(args,rhs,sens_rhs,args["training_size"])

    NN["encoder"] = encoder |> gpu
    NN["decoder"] = decoder |> gpu
    NN["par_decoder"] = par_decoder |> gpu 
    NN["par_encoder"] = par_encoder |> gpu
    
    training_data["x"] = x_train
    training_data["dxdt"] = dx_train
    training_data["alpha"] = alpha_train
    training_data["dxda"] = dxda_train
    training_data["dtdxda"] = dtdxda_train
    
    return NN, training_data
end

end
