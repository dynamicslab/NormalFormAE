ENV["JULIA_CUDA_VERBOSE"] = true
ENV["JULIA_CUDA_MEMORY_POOL"] = "split" # Efficient allocation to GPU (Julia garbage collection is inefficient for this code apparently)
ENV["JULIA_CUDA_MEMORY_LIMIT"] = 8000_000_000

import Pkg
Pkg.activate(".")
using NormalFormAE
using Zygote
using Plots, Flux, LaTeXStrings

# Load correct rhs
include("../src/problems/scalar.jl")

args = Dict()

# Define all necessary parameters
args["ExpName"] = "LP"
args["par_dim"] = 1
args["z_dim"] = 1
args["x_dim"] = 1
args["mean_init"] = [0.001f0] .+ zeros(Float32,args["x_dim"])
args["mean_a"] = [0.0f0]
args["xVar"] = 0.1f0
args["aVar"] = 1f0
args["tspan"] = [0.0, 0.0008]
args["tsize"] = 50

args["x_lp"] = -60.0f0
args["p_lp"] = -60.0f0
args["p_pf"] = 30.0f0
args["p_tc"] = args["p_lp"] - args["x_lp"]^2

if args["ExpName"] == "LP"
    args["bif_x"] = args["x_lp"]
    args["bif_p"] = args["p_lp"]
    dzdt_rhs = dzdt_lp_rhs
    dzdt_solve = dzdt_lp_solve
    println("Problem: Limit point")
    
elseif args["ExpName"] == "Pitch"
    args["bif_x"] = 0.0f0
    args["bif_p"] = args["p_pf"]
    dzdt_rhs = dzdt_pitchfork_rhs
    dzdt_solve = dzdt_pitchfork_solve
    println("Problem: Pitchfork")
else
    args["bif_x"] = 0.0f0
    args["bif_p"] = args["p_tc"]
    dzdt_rhs = dzdt_trans_rhs
    dzdt_solve = dzdt_trans_solve
    println("Problem: Transcritical")
end

args["AE_widths"] = [1,20,20,1]
args["AE_acts"] = ["elu","elu","id"]
args["Par_widths"] = [1,10,10,1]
args["Par_acts"] = ["elu","elu","id"]

args["training_size"] = 500
args["test_size"] = 20
args["BatchSize"] = 100

args["nPlots"] = 5
args["nEnsPlot"] = 5
args["varPlot"] = 0.1f0

args["tscale_init"] = 0.05f0

# Generate training data, test data and all neural nets
for i=5:10
    NN, training_data, test_data = pre_train(args,dxdt_rhs,dxdt_sens_rhs)
    NN_old = NN
    training_data_old = training_data
    test_data_old = test_data
    # ## Train sequence 1 (bad)
    args["nEpochs"] = 100
    args["nIt"] = 1
    args["nIt_tscale"] = 1
    args["ADAMarg"] = 0.001
    args["P_AE_state"] = 1f0
    # args["P_cons_x"] = 0.0001f0
    args["P_cons_x"] = 0.001f0
    args["P_cons_z"] = 0.001f0
    args["P_AE_par"] = 1f0
    args["P_sens_dtdzdb"] = 0.0f0
    args["P_sens_x"] = 0.0f0
    args["P_AE_id"] = 0.0f0
    #args["P_NLRAN_in"] = 0.001f0
    #args["P_NLRAN_out"] = 0.001f0
    args["P_NLRAN_in"] = 0.01f0
    args["P_NLRAN_out"] = 0.01f0
    args["P_orient"] = 1.0f0
    args["P_zero"] = 1.0f0
    args["pre_train"] = true
    trained_NN = (NN["encoder"],NN["decoder"],NN["par_encoder"],NN["par_decoder"])
    # Bad training
    args["P_NLRAN_in"] =0
    args["P_NLRAN_out"] = 0
    args["nIt_tscale"] = 0
    plot_ = train(args,training_data,test_data,NN,trained_NN,dzdt_rhs,dzdt_solve,dzdt_sens_rhs)
    savefig("Trials-1509/lp_bad-$(i).pdf")
    plot(1:1000,test_data["x"]',label="Original system")
    plot!(1:1000,NN["encoder"](gpu(test_data["x"]))',label="Learned latent dynamics",title=latexstring("\\textrm{System: LP, Test data, } \\tau = $(NN["tscale"].^2)  "))
    savefig("Trials-1509/lp_bad_orig_vs_trained-$(i).pdf")
    Plots.closeall()

    # ## Train sequence 2 (good)


    args["nEpochs"] = 100
    args["nIt"] = 1
    args["nIt_tscale"] = 1
    args["ADAMarg"] = 0.001
    args["P_AE_state"] = 1f0
    # args["P_cons_x"] = 0.0001f0
    args["P_cons_x"] = 0.001f0
    args["P_cons_z"] = 0.001f0
    args["P_AE_par"] = 1f0
    args["P_sens_dtdzdb"] = 0.0f0
    args["P_sens_x"] = 0.0f0
    args["P_AE_id"] = 0.0f0
    #args["P_NLRAN_in"] = 0.001f0
    #args["P_NLRAN_out"] = 0.001f0
    args["P_NLRAN_in"] = 0.01f0
    args["P_NLRAN_out"] = 0.01f0
    args["P_orient"] = 1.0f0
    args["P_zero"] = 1.0f0
    args["pre_train"] = false

    NN = NN_old
    training_data = training_data_old
    test_data = test_data_old
    trained_NN = (NN["encoder"],NN["decoder"],NN["par_encoder"],NN["par_decoder"])
    # Train 1
    args["nEpochs"] = 5
    plot_ = train(args,training_data,test_data,NN,trained_NN,dzdt_rhs,dzdt_solve,dzdt_sens_rhs)

    # Train 2
    args["nEpochs"] = 50
    args["P_NLRAN_in"] = 0
    args["P_NLRAN_out"] = 0
    args["pre_train"] = false
    plot_ = train(args,training_data,test_data,NN,trained_NN,dzdt_rhs,dzdt_solve,dzdt_sens_rhs)
    savefig("Trials-1509/lp_good-$(i).pdf")
    plot(1:1000,test_data["x"]',label="Original system")
    plot!(1:1000,NN["encoder"](gpu(test_data["x"]))',label="Learned latent dynamics",title=latexstring("\\textrm{System: LP, Test data, } \\tau = $(NN["tscale"].^2)  "))
    savefig("Trials-1509/lp_good_orig_vs_train-$(i).pdf")
end
