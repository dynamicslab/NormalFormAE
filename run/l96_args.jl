args = Dict()
args["ExpName"] = "l96"
dir_ = "/home/manu/Documents/Work/NormalFormAE"
args["DataDir"] = "/home/manu/Documents/Work/NormalFormAEData"

# Dimension parameters
args["par_dim"] = 1
args["z_dim"] = 2
args["x_dim"] = 64

# Neural field parameters
args["bif_x"] = 0.84975f0 .+ zeros(Float32, args["x_dim"]) # Set this to 0 to avoid translation in ground truth data
args["bif_p"] = 0.84975f0 #   Set this to 0 to avoid translation in ground truth data

# Data generation parameters
args["mean_init"] = 0.01f0 .+ zeros(Float32,args["x_dim"])
args["mean_a"] = [0.0]
args["xVar"] = 0.1f0
args["aVar"] = 0.5f0
args["tspan"] = [0.0, 40.0]
args["tsize"] = 200

# Neural net parameters
args["AE_widths"] = [64,32,16,2]
args["AE_acts"] = ["elu","elu","id"]
args["Par_widths"] = [1,16,16,1]
args["Par_acts"] = ["elu","elu","id"]
args["tscale_init"] = 1f0

# Dataset generation parameters
args["training_size"] = 1000
args["BatchSize"] = 100
args["test_size"] = 20

# Set arguments
args["nEpochs"] = 100
args["nIt"] = 1
args["nIt_tscale"] = 1
args["ADAMarg"] = 0.001
args["P_AE_state"] = 1f0
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
args["nPlots"] = 10 # How many test data plots
args["nEnsPlot"] = 5 # How many in gray ensemble
args["varPlot"] = 0.1f0 # Variance of initial values
