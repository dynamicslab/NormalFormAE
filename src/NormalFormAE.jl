module NormalFormAE

using DifferentialEquations, Flux, Distributions, Plots, CUDAapi, Random, PolyChaos, Zygote

# if has_cuda()
#     @info "CUDA is on"
#     import CuArrays
#     CuArrays.allowscalar(false)
# end

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
    
    encoder, decoder, hom_encoder, hom_decoder = get_autoencoder(args) 
    t_train,z_train,dz_train,x_train,dx_train = gen(args,rhs,args["training_size"],2)
    t_test,z_test,dz_test,x_test,dx_test = gen(args,rhs,args["test_size"],2,"test")

    NN["encoder"] = encoder |> gpu
    NN["decoder"] = decoder |> gpu
    NN["hom_decoder"] = hom_decoder |> gpu 
    NN["hom_encoder"] = hom_encoder |> gpu
    
    training_data["t"] = t_train |> gpu
    training_data["z"] = z_train |> gpu
    training_data["dz"] = dz_train |>gpu
    training_data["x"] = x_train |> gpu
    training_data["dx"] = dx_train |> gpu
    
    test_data["t"] = t_test |> gpu
    test_data["z"] = z_test |> gpu
    test_data["dz"] = dz_test |> gpu
    test_data["x"] = x_test |> gpu
    test_data["dx"] = dx_test |> gpu

    return NN, training_data, test_data
end

end
