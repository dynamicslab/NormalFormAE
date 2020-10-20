ENV["JULIA_CUDA_VERBOSE"] = true
ENV["JULIA_CUDA_MEMORY_POOL"] = "split" # Efficient allocation to GPU (Julia garbage collection is inefficient for this code apparently)
ENV["JULIA_CUDA_MEMORY_LIMIT"] = 8000_000_000

import Pkg
Pkg.activate(".")
using NormalFormAE
using Zygote

# Load correct rhs
include("../src/problems/Brusselator_PDE.jl")

args = Dict()

# Define all necessary parameters
args["ExpName"] = "Brusselator"
args["par_dim"] = 1
args["z_dim"] = 2
args["x_dim"] = 64
args["mean_init"] = 0.01f0 .+ zeros(Float32,args["x_dim"])
args["mean_a"] = [0.0]
args["xVar"] = 1.0f0
args["aVar"] = 1f0
args["tspan"] = [0.0, 0.001]
args["tsize"] = 500

args["par_aa"] = 2.0f0
args["par_bb"] = 5.45f0
args["par_dx"] = 0.008f0
args["par_dy"] = 0.004f0

bif_x = zeros(Float32,args["x_dim"])
i = 1
while i<args["x_dim"]
    global i
    bif_x[i] = args["par_aa"]
    bif_x[i+1] = args["par_bb"]/args["par_aa"]
    i = i+2
end

args["bif_x"] = bif_x
args["bif_p"] = [0.5130]

args["AE_widths"] = [64,32,16,2]
args["AE_acts"] = ["sigmoid","sigmoid","id"]
args["Par_widths"] = [1,16,16,1]
args["Par_acts"] = ["elu","elu","id"]

args["training_size"] = 1000
args["BatchSize"] = 100
args["test_size"] = 20

args["nPlots"] = 10
args["nEnsPlot"] = 5
args["varPlot"] = 0.1f0

args["tscale_init"] = 1f0

# Generate training data, test data and all neural nets
NN, training_data = pre_train(args,dxdt_rhs,dxdt_sens_rhs)

## Train sequence 1
args["nEpochs"] = 10
#args["nBatches"] = 100
args["nIt"] = 1
args["nIt_tscale"] = 0
args["ADAMarg"] = 0.001
args["P_AE_state"] = 1f0
args["P_cons_x"] = 0.001f0
args["P_cons_z"] = 0.001f0
args["P_AE_par"] = 1f0
args["P_sens_dtdzdb"] = 0.0f0
args["P_sens_x"] = 0.0f0
args["P_AE_id"] = 0.01f0
args["P_NLRAN_in"] = 0
args["P_NLRAN_out"] = 0
args["P_orient"] = 1.0f0
args["P_zero"] = 1.0f0
args["pre_train"] = true
trained_NN = (NN["encoder"],NN["decoder"],NN["par_encoder"],NN["par_decoder"])
#plot_ = train(args,training_data,test_data,NN,trained_NN,dzdt_rhs,dzdt_solve,dzdt_sens_rhs)

