ENV["JULIA_CUDA_VERBOSE"] = true
ENV["JULIA_CUDA_MEMORY_POOL"] = "split" # Efficient allocation to GPU (Julia garbage collection is inefficient for this code apparently)
ENV["JULIA_CUDA_MEMORY_LIMIT"] = 7500_000_000

import Pkg
Pkg.activate(".")
using NormalFormAE
using CuArrays
using Flux, Plots, ArgParse
#CuArrays.allowscalar(false)

#------------Argument Parsing-------------------------
args = ArgParseSettings()
@add_arg_table args begin
    "--ExpName"
    help = "Experiment name"
    arg_type = String
    default = "Test"
    "--model"
    help = "Choose ODE model"
    arg_type = String
    default = "LP"
    "--Kathleen"
    action = :store_true
    "--NoiseVar"
    arg_type = Float64
    default = 2.0
    "--normalize"
    arg_type = Float32
    default = 1.0f0
    "--p_normalize"
    arg_type = Float32
    default = 1.0f0
    "--training_size"
    arg_type = Int64
    default = 5000
    "--test_size"
    arg_type = Int64
    default = 20
    "--nEpochs"
    arg_type = Int64
    default = 20
    "--nBatches"
    arg_type = Int64
    default = 100
    "--nIterations"
    arg_type = Int64
    default = 50
    "--ADAMarg"
    arg_type = Float64
    default = 0.01
    "--P_DataFid"
    arg_type = Float32
    default = 1.0f0
    "--P_Hom"
    arg_type = Float32
    default = 1.0f0
    "--P_dx"
    arg_type = Float32
    default = 0.001f0
    "--P_dz"
    arg_type = Float32
    default = 0.001f0
    "--P_par"
    arg_type = Float32
    default = 0.01f0
    "--AE_widths"
    arg_type = Int64
    nargs = '+'
    default = [50,25,12,2]
    "--AE_acts"
    arg_type = String
    nargs = '+'
    default = ["leakyrelu","leakyrelu","id"]
    "--Par_widths"
    arg_type = Int64
    nargs = '+'
    default = [2,10,10,2]
    "--Par_acts"
    arg_type = String
    nargs = '+'
    default = ["leakyrelu","leakyrelu","id"]
    "--z_dim"
    arg_type = Int64
    default = 2
    "--par_dim"
    arg_type = Int64
    default = 1
    "--expansion_order"
    arg_type = Int64
    default = 1
    "--x_spatial_scale"
    arg_type = Int64
    default = 50
    "--alpha_spatial_scale"
    arg_type = Int64
    default = 5
    "--tspan"
    arg_type = Float64
    nargs = '+'
    default = [0.0,0.1]
    "--tsize"
    arg_type = Int64
    default = 100
    "--mean_init"
    nargs = '+'
    arg_type = Float64
    default = [0.01,0.0]
end
parsed_args = parse_args(args)

if parsed_args["model"] == "LP"
    include("src/models/LP.jl")
elseif parsed_args["model"] == "Hopf"
    include("src/models/Hopf.jl")
elseif parsed_args["model"] == "Lorenz"
    include("src/models/Lorenz.jl")
end

println("Module created.")

NN, training_data, test_data = pre_train(parsed_args,rhs)

println("NN constructed, data obtained.")

train(parsed_args,training_data,test_data, NN, rhs)    
    
    
            
