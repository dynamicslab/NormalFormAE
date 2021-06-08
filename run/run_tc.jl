
ENV["GKSwstype"] = "nul"
using Pkg
Pkg.activate("/home/kaliam/NormalFormAE/.")
using NormalFormAE, Flux,CUDA, Zygote, Plots, DifferentialEquations, FileIO, Printf
device!(CUDA.CuDevice(0))
include("/home/kaliam/NormalFormAE/problems/tc.jl")
#include("/home/kaliam/NormalFormAE/run/run_nf_old.jl")

x_model_name = :toy
z_model_name = :tc
x_dim = 1
par_dim = 1
z_dim = 1
tsize = 250 # previously 500 
tspan = [0.0,5.0] # previously 200i
xVar = 0.1
aVar = 0.1
#mean_ic_x = (Array{Float32}(range(0,500,length=500))./500f0.*aVar)' 
mean_ic_x = [0.0]
mean_ic_a = [0.0]
x_rhs = dxdt_rhs
x_solve = dxdt_solve
machine = cpu
 

model_x = xModel(x_model_name, x_dim, par_dim, tsize,tspan,xVar,aVar,mean_ic_x,mean_ic_a,x_rhs,x_solve, args)
model_z = NormalForm(:tc,z_dim ,par_dim, dzdt_rhs, dzdt_solve)

state = AE(:State, 1,1, [20,20,20],:elu,machine)
par = AE(:Par, 1,1,[10,10],:elu,machine)

tscale_init = [0.1f0] |> machine

training_size = 500
test_size = 20


data_dir = nothing

P_reg = [0f0, 1f0, 0.0f0, 0.001f0, 0.001f0, 0f0, 0.0f0]

# data_dir = "/home/kaliam/NFAEdata"

nfae = NFAE(:toy, :tc, model_x, model_z, training_size, test_size, state, par, nothing, tscale_init,
                       P_reg,machine, 20,20,0.5,data_dir)

#load_data(nfae)
#load_params(nfae)
nfae.data_dir = "/home/kaliam/NFAEdata/"
load_params(nfae)
# load_data(nfae)


# Sort test data by order of parameters
ind_ = sortperm(nfae.test_data["alpha"][1,:])
nfae.test_data["alpha"] = sort(nfae.test_data["alpha"],dims=2)
nfae.test_data["x"] = nfae.test_data["x"][:,:,ind_]
nfae.test_data["dx"] = nfae.test_data["dx"][:,:,ind_]


# Load test aata
x_test = reduce(hcat,[nfae.test_data["x"][:,:,i]./0.1f0 for i in 1:nfae.test_size]) |> nfae.machine
dx_test = reduce(hcat,[nfae.test_data["dx"][:,:,i]./0.1f0 for i in 1:nfae.test_size]) |> nfae.machine
alpha_test = nfae.test_data["alpha"]./0.5f0 |> nfae.machine
nfae.training_data["alpha"] = nfae.training_data["alpha"]./0.5f0

nfae.training_data["x"] = nfae.training_data["x"]./0.1f0
nfae.training_data["dx"] = nfae.training_data["dx"]./0.1f0
nfae.test_data["x"] = nfae.test_data["x"]
nfae.test_data["dx"] = nfae.test_data["dx"]

nEpochs = 0
batchsize = 20 # before 50
ctr = 1
p = gen_plot(nfae.model_z.z_dim, nfae.nPlots)
adamarg = 0.001
opt = ADAM(adamarg)

# Hetcers, should go into nfae at some point
loss_train_full = []
loss_test_full = []
last_loss = 1e10
loss_test = 0.0f0
rel_loss_test = 1f0

include("/home/kaliam/NormalFormAE/src/utils/pretrain.jl")
include("/home/kaliam/NormalFormAE/src/utils/train.jl")

if nfae.state.act == "id"
    act_ = nothing
else
    act_ = Symbol(nfae.state.act)
end

while ctr < nEpochs*(training_size/batchsize) 
   global adamarg
   ctr = 1 
    nfae.state = AE(:State, nfae.state.in_dim,nfae.state.out_dim,nfae.state.widths,act_,nfae.machine) # respawn
    nfae.par = AE(:Par, nfae.par.in_dim,nfae.par.out_dim,nfae.par.widths,act_,nfae.machine) # respawn
  # pretrain(10000,batchsize,5e-2,ADAM(0.001))
   load_params(nfae) 
   train(nfae,nEpochs,batchsize, x_test, dx_test, alpha_test)
   adamarg = 0.001
end
