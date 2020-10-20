args = Dict()
args["ExpName"] = "pf"
dir_ = "/home/manu/Documents/Work/NormalFormAE"
args["DataDir"] = "/home/manu/Documents/Work/NormalFormAEData"

# Dimension parameters
args["par_dim"] = 1
args["z_dim"] = 1
args["x_dim"] = 1

# 1D sysrtem parameters
args["x_lp"] = -60.0f0
args["p_lp"] = -60.0f0
args["p_pf"] = 30.0f0
args["p_tc"] = args["p_lp"] - args["x_lp"]^2
args["bif_x"] = 0.0f0
args["bif_p"] = args["p_pf"]

# Data generation parameters
args["mean_init"] = [0.001f0] .+ zeros(Float32,args["x_dim"])
args["mean_a"] = [0.0f0]
args["xVar"] = 0.1f0
args["aVar"] = 1f0
args["tspan"] = [0.0, 0.5]
args["tsize"] = 50

# NN parameters
args["AE_widths"] = [1,20,20,1]
args["AE_acts"] = ["elu","elu","id"]
args["Par_widths"] = [1,10,10,1]
args["Par_acts"] = ["elu","elu","id"]
args["tscale_init"] = 0.1f0

# Dataset generation parameters
args["training_size"] = 500
args["test_size"] = 20
args["BatchSize"] = 100

# Set arguments
args["nEpochs"] = 200
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
args["P_NLRAN_in"] = 0
args["P_NLRAN_out"] = 0
args["P_orient"] = 1.0f0
args["P_zero"] = 1.0f0

# Plotting parameters
args["nPlots"] = 10
args["nEnsPlot"] = 5
args["VarPlot"] = 0.1f0
