ENV["JULIA_CUDA_VERBOSE"] = true
ENV["JULIA_CUDA_MEMORY_POOL"] = "split" # Efficient allocation to GPU (Julia garbage collection is inefficient for this code apparently)
ENV["JULIA_CUDA_MEMORY_LIMIT"] = 8000_000_000

import Pkg
Pkg.activate(".")
using NormalFormAE, CUDAapi
if has_cuda()
    @info "CUDA is on"
    import CuArrays
    CuArrays.allowscalar(false)
end

# Load correct rhs
include("../src/models/FhN.jl")

args = Dict()

# Define all necessary parameters
args["ExpName"] = "FhN_tanh"
args["par_dim"] = 4
args["z_dim"] = 2
args["StateVar"] = 0.1f0
args["ParVar"] = 0.1f0
args["AE_widths"] = [64,32,16,2]
args["AE_acts"] = ["sigmoid","sigmoid","id"]
args["Par_widths"] = [10,8,4]
args["Par_acts"] = ["sigmoid","id"]
args["expansion_order"] = 1
args["x_spatial_scale"] = 64
args["alpha_spatial_scale"] = 10
args["training_size"] = 5000
args["test_size"] = 20
args["tspan"] = [0.0,0.01]
args["tsize"] = 200
args["mean_init"] = [-1.0, 1.0, 3.0, 0.06667,0.3333,0.06667]
args["normalize"] = 5.0f0
args["p_normalize"] = 5.0f0

# Generate training data, test data and all neural nets
NN, training_data, test_data = pre_train(args,rhs)

## Train sequence 1
args["nEpochs"] = 1
args["nBatches"] = 100
args["nIterations"] = 1
args["ADAMarg"] = 0.0001
args["P_DataFid"] = 1f0
args["P_dx"] = 0.1f0
args["P_dz"] = 0.001f0
args["P_par"] = 1.0f0
trained_NN = (NN["encoder"],NN["decoder"],NN["par_encoder"],NN["par_decoder"])
train(args,training_data,test_data,NN,trained_NN,rhs)

args["nEpochs"] = 1
args["nBatches"] = 1000
args["nIterations"] = 1
args["ADAMarg"] = 0.0001
args["P_DataFid"] = 1.0f0
args["P_dx"] = 0.1f0
args["P_dz"] = 0.1f0
args["P_par"] = 1.0f0
trained_NN = (NN["par_encoder"],NN["par_decoder"],NN["encoder"],NN["decoder"])
train(args,training_data,test_data,NN,trained_NN,rhs)



