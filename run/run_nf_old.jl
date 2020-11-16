args = Dict()
args["ExpName"] = "nf"
dir_ = "/home/manu/Documents/Work/NormalFormAE"
args["DataDir"] = "/home/manu/Documents/Work/NormalFormAEData"

using FFTW, DifferentialEquations

# Dimension parameters
args["par_dim"] = 1
args["z_dim"] = 2
args["x_dim"] = 128

# Neural field parameters
Nx = args["x_dim"]/2
S = 1
period=12
dx = period/Nx
x = dx .* Array(-Nx/2:(Nx/2-1))
sig = 1.2
we = 1
sige = 1
wi = 0
sigi = 1
args["gain"] = 2.75
args["theta"] = 0.375
args["tau"] = 10.0
args["beta"] = 6.0f0
ww = we/sige/sqrt(pi) .* exp.(-(x ./ sige).^2) .- wi/sigi/sqrt(pi) .* exp.(-(x ./ sigi).^2)
args["ft_conn"] = fft(fftshift(dx .* ww))
I0 = 0.8040 # Bifurcation point
args["II"] = I0*exp.(-(x./sig).^2)
y0 = [0.2*args["II"]; 0.3*args["II"]]

args["bif_x"] = 0.0 # Set this to 0 to avoid translation in ground truth data
args["bif_p"] = 0.0  # Set this to 0 to avoid translation in ground truth data
x_solve_old(du,u, par, t) = dxdt_solve(args,du,u,par,t)
prob  = ODEProblem(x_solve_old, y0, (0.0f0,100.0f0),(I0,))
sol = solve(prob,Tsit5(),saveat=Array(0.0:0.05:100.0),reltol = 1e-8,dtmax = 0.1);
args["bif_x"] = Array(sol)[:,end]; # Now set fixed translation (which is wrong)
args["bif_p"] = 0.8040 # Fixed trnslation (wrong)

# Data generation parameters
args["mean_init"] = 0.01f0 .+ zeros(Float32,args["x_dim"])
args["mean_a"] = [0.0]
args["xVar"] = 0.1f0
args["aVar"] = 0.5f0
args["tspan"] = [0.0, 200]
args["tsize"] = 500

# Neural net parameters
args["AE_widths"] = [128,64,32,16,2]
args["AE_acts"] = ["elu","elu","elu","id"]
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
