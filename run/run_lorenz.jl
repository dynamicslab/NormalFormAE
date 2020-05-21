ENV["JULIA_CUDA_VERBOSE"] = true
ENV["JULIA_CUDA_MEMORY_POOL"] = "split" # Efficient allocation to GPU (Julia garbage collection is inefficient for this code apparently)
ENV["JULIA_CUDA_MEMORY_LIMIT"] = 8000_000_000

import Pkg
Pkg.activate(".")
using NormalFormAE

# Load correct rhs
include("../src/models/Lorenz.jl")

args = Dict()

# Define all necessary parameters
args["ExpName"] = "Lorenz_tanh"
args["par_dim"] = 2
args["z_dim"] = 3
args["StateVar"] = 2.0f0
args["ParVar"] = 0.0f0
args["AE_widths"] = [128,64,32,3]
args["AE_acts"] = ["tanh","tanh","tanh"]
args["Par_widths"] = [10,8,5,2]
args["Par_acts"] = ["tanh","tanh","tanh"]
args["expansion_order"] = 3
args["x_spatial_scale"] = 128
args["alpha_spatial_scale"] = 10
args["training_size"] = 10000
args["test_size"] = 20
args["tspan"] = [0.0,5.0]
args["tsize"] = 200
args["mean_init"] = [0.0, 0.0, 25.0, 10.0, 28.0]
args["normalize"] = 50.0f0
args["p_normalize"] = 40.0f0

# Generate training data, test data and all neural nets
NN, training_data, test_data = pre_train(args,rhs)

## Train sequence 1
args["nEpochs"] = 3
args["nBatches"] = 5000
args["nIterations"] = 10
args["ADAMarg"] = 0.001
args["P_DataFid"] = 0.001f0
args["P_dx"] = 1f0
args["P_dz"] = 0.1f0
args["P_par"] = 1.0f0
trained_NN = (NN["encoder"],NN["decoder"])
train(args,training_data,test_data,NN,trained_NN,rhs)

args["nEpochs"] = 1
args["nBatches"] = 1000
args["nIterations"] = 1
args["ADAMarg"] = 0.001
args["P_DataFid"] = 1.0f0
args["P_dx"] = 0.1f0
args["P_dz"] = 0.0001f0
args["P_par"] = 1.0f0
trained_NN = (NN["par_encoder"],NN["par_decoder"],NN["encoder"],NN["decoder"])
train(args,training_data,test_data,NN,trained_NN,rhs)



## BiLevel_Lorenz_1905
# ## Train sequence 1
# args["nEpochs"] = 1
# args["nBatches"] = 1000
# args["nIterations"] = 1
# args["ADAMarg"] = 0.001
# args["P_DataFid"] = 1.0f0
# args["P_dx"] = 0.1f0
# args["P_dz"] = 0.0001f0
# args["P_par"] = 1.0f0
# trained_NN = (NN["encoder"],NN["decoder"])
# train(args,training_data,test_data,NN,trained_NN,rhs)

# args["nEpochs"] = 1
# args["nBatches"] = 1000
# args["nIterations"] = 10
# args["ADAMarg"] = 0.001
# args["P_DataFid"] = 0.0f0
# args["P_dx"] = 0.1f0
# args["P_dz"] = 0.0001f0
# args["P_par"] = 1.0f0
# trained_NN = (NN["par_encoder"],NN["par_decoder"])
# train(args,training_data,test_data,NN,trained_NN,rhs)



