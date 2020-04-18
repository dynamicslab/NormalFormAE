import Pkg
Pkg.activate(".")
using NormalFormAE
using Flux, Plots, ArgParse
using CuArrays
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
    "--training_size"
    arg_type = Int64
    default = 1000
    "--test_size"
    arg_type = Int64
    default = 20
    "--nEpochs"
    arg_type = Int64
    default = 20
    "--nBatches"
    arg_type = Int64
    default = 5
    "--nIterations"
    arg_type = Int64
    default = 100
    "--ADAMarg"
    arg_type = Float64
    default = 0.001
    "--P_DataFid"
    arg_type = Float32
    default = 1.0f0
    "--P_Hom"
    arg_type = Float32
    default = 1.0f0
    "--P_dx"
    arg_type = Float32
    default = 1.0f0
    "--P_dz"
    arg_type = Float32
    default = 1.0f0
    "--AE_widths"
    arg_type = Int64
    nargs = '+'
    default = [128,64,32,16,2]
    "--AE_acts"
    arg_type = String
    nargs = '+'
    default = ["sigmoid","sigmoid","sigmoid","id"]
    "--Hom_widths"
    arg_type = Int64
    nargs = '+'
    default = [2,2,2]
    "--Hom_acts"
    arg_type = String
    nargs = '+'
    default = ["sigmoid","id"]
    "--z_dim"
    arg_type = Int64
    default = 2
    "--expansion_order"
    arg_type = Int64
    default = 1
    "--spatial_scale"
    arg_type = Int64
    default = 128
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
end

println("Module created.")

NN, training_data, test_data = pre_train(parsed_args,rhs)

println("NN constructed, data obtained.")

train(parsed_args,training_data,test_data, NN, rhs)    
    
    
            
