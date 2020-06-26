ENV["JULIA_CUDA_VERBOSE"] = true
ENV["JULIA_CUDA_MEMORY_POOL"] = "split" # Efficient allocation to GPU (Julia garbage collection is inefficient for this code apparently)
ENV["JULIA_CUDA_MEMORY_LIMIT"] = 8000_000_000

import Pkg
Pkg.activate(".")
using NormalFormAE
using Zygote

# Load correct rhs
include("../src/models/Hopf.jl")

args = Dict()

# Define all necessary parameters
args["ExpName"] = "Hopf"
args["par_dim"] = 1
args["z_dim"] = 2
args["StateVar"] = 2.0f0
args["ParVar"] = 2.0f0
args["AE_widths"] = [64,32,16,2]
args["AE_acts"] = ["sigmoid","sigmoid","id"]
args["Par_widths"] = [1,5,5,1]
args["Par_acts"] = ["elu","elu","id"]
args["expansion_order"] = 1
args["x_spatial_scale"] = 64
args["alpha_spatial_scale"] = 5
args["training_size"] = 1000
args["test_size"] = 20
args["tspan"] = [0.0,20.0]
args["tsize"] = 100
args["mean_init"] = [0.01, 0.01, 0.01]
args["normalize"] = 1.0f0
args["p_normalize"] = 1.0f0

# Generate training data, test data and all neural nets
NN, training_data, test_data = pre_train(args,rhs)

## Train sequence 1
args["nEpochs"] = 1
#args["nBatches"] = 100
args["nIt_AE"] = 3
args["nIt_u0"] = 10
args["nIt_par"] = 3
args["ADAMarg"] = 0.01
args["P_DataFid"] = 1f0
args["P_dx"] = 0.001f0
args["P_dz"] = 0.001f0
args["P_par"] = 1f0
args["P_cons"] = 0.001f0
args["P_dec2"] = 1.0f0
trained_NN1 = (NN["encoder"],NN["decoder"])
trained_NN2 =  (NN["par_encoder"],NN["par_decoder"])
train(args,training_data,test_data,NN,trained_NN1,trained_NN2,rhs,rhs_solve)

# args["nEpochs"] = 1
# args["nBatches"] = 1000
# args["nIterations"] = 1
# args["ADAMarg"] = 0.001
# args["P_DataFid"] = 1.0f0
# args["P_dx"] = 0.1f0
# args["P_dz"] = 0.0001f0
# args["P_par"] = 1.0f0
# trained_NN = (NN["par_encoder"],NN["par_decoder"],NN["encoder"],NN["decoder"])
# train(args,training_data,test_data,NN,trained_NN,rhs)


