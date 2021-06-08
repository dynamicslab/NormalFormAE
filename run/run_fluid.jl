ENV["GKSwstype"] = "nul"
using Pkg
Pkg.activate("/home/kaliam/NormalFormAE/.")
using CUDA
using LinearAlgebra
device!(CUDA.CuDevice(2))

using NormalFormAE, Flux, Zygote, Plots, DifferentialEquations, IJulia, BSON, JLD2, FileIO,Random,Printf
include("/home/kaliam/NormalFormAE/problems/nf.jl")
include("/home/kaliam/NormalFormAE/run/run_nf_old.jl")

# This is the experiment for the fluid flow model. Data was obtained by simulating the fluid flow past
# a cylinder example in ViscousFlows.jl. Time derivatives are obtained numerically from data, using
# first order finite differences. Check data generation files for precise spatial details.
# The bifurcation (we estimate from data, to make this exampple data-driven) occurs at Re = 44.6.

# dt = 0.125
# Scaling for alpha = 80
# Old parameters stored in fluid_pod-Hopf/encoder.bson

x_model_name = :fluid
z_model_name = :Hopf
x_dim = 4
par_dim = 1
z_dim = 2
tsize = 293
tspan = [0.0,10.0]
xVar = 0.1
aVar = 0.5
mean_ic_x = [0.0] 
mean_ic_a = [0.0]
x_rhs = dxdt_rhs
x_solve = dxdt_solve
machine = gpu # gpu/cpu

model_x = xModel(x_model_name, x_dim, par_dim, tsize,tspan,xVar,aVar,mean_ic_x,mean_ic_a,x_rhs,x_solve, args) # Redundant for the fluid example, but needed for nfae
model_z = NormalForm(:Hopf,z_dim ,par_dim, dzdt_rhs, dzdt_solve)

state = AE(:State, x_dim,z_dim, [30,30],:elu,machine)
par = AE(:Par, 1,1,[10,10],:elu,machine)
trans = AE(:Trans,1,2,[16,16],:elu,machine)

tscale_init = [0.6f0] |> machine

training_size = 220
test_size = 19

data_dir = "/home/kaliam/NFAEdata/"

P_reg = [0.0f0, 1.0f0, 1.0f0, 0.0001f0, 0.0001f0, 0f0, 0.1f0]

nfae = NFAE(:fluid, :Hopf, model_x, model_z, training_size, test_size, state, par, nothing, tscale_init,
            P_reg,machine, 10,20,0.1,data_dir)   

nfae.model_x.tsize = 293
nfae.model_x.x_dim = 4

## TRAINING

#----- Load test and training data ------
shuffle_ind = shuffle(1:239)
test_ind = sort(shuffle_ind[end-18:end])
# test_ind = [12,14,20,23,28,29,41,58,65,59,96,122,132,179,190,193,198,200,211]
train_ind = shuffle_ind[1:220]



train_ind = FileIO.load("/home/kaliam/NFAEdata/fluid-Hopf/train_ind.jld2","train_ind")
test_ind = FileIO.load("/home/kaliam/NFAEdata/fluid-Hopf/test_ind.jld2","test_ind")

# Load

fluid_data = FileIO.load("/home/kaliam/NFAEdata/fluid-Hopf/fluid_data.jld2","fluid_data")
drop = [[1.0f0 0.0f0 1.0f0 0.0f0];[0.0f0 1.0f0 0.0f0 1.0f0]] 

nfae.training_data = Dict()
nfae.test_data = Dict()

for i in ["x", "dx","alpha"]
    if i != "alpha"
        nfae.training_data[i] = fluid_data[i][1:nfae.model_x.x_dim,:,train_ind] |> nfae.machine
        nfae.test_data[i] = fluid_data[i][1:nfae.model_x.x_dim,:,test_ind] |> nfae.machine
    else
        nfae.training_data[i] = reshape(fluid_data[i][train_ind]./80.0f0 .- 44.6f0/80.0f0,1,220) |> nfae.machine
        nfae.test_data[i] = reshape(fluid_data[i][test_ind]./80.0f0 .- 44.6f0/80.0f0,1,19) |> nfae.machine
    end
end


x_test = cat([nfae.test_data["x"][:,:,i] for i in 1:19]...,dims=2) |> nfae.machine
dx_test = cat([nfae.test_data["dx"][:,:,i] for i in 1:19]...,dims=2) |> nfae.machine
alpha_test = nfae.test_data["alpha"] |> nfae.machine

nfae.model_x.x_dim = 4
drop = rand(4,4)
s = svd(drop)
drop = s.U*s.Vt |> nfae.machine
lift_mat = inv(cpu(drop))
lift(x) = x

drop = FileIO.load("/home/kaliam/NFAEdata/fluid-Hopf/drop.jld2","drop") |> nfae.machine
lift_mat = inv(cpu(drop))


# --------- Manually set up neural networks --------
nfae.par = AE(:Par, 1,1,[10,10],:tanh,machine)
nfae.state = AE(:State, nfae.model_x.x_dim,nfae.model_z.z_dim,[20,20,20],:tanh,machine) # before [20,20]

load_params(nfae)

#------------- Training ---------

nEpochs =0
batchsize = 110
ctr = 1
p = gen_plot(nfae.model_z.z_dim, nfae.nPlots)
adamarg = 0.001
opt = ADAM(adamarg)

# Helpers, should go into nfae at some point
loss_train_full = []
loss_test_full = []
last_loss = 1e10
loss_test = 0.0f0
rel_loss_test = 1f0

include("/home/kaliam/NormalFormAE/src/utils/pretrain_fluid.jl")
include("/home/kaliam/NormalFormAE/src/utils/train_fluid.jl")

if nfae.state.act == "id"
    act_ = nothing
else
    act_ = Symbol(nfae.state.act)
end
#while true 
#   global adamarg
#   global loss_train_full, loss_test_full, drop, lift_mat
#
#   loss_train_full = []
#   loss_test_full = []
#   while minimum(abs.(drop)) < 0.1f0
#        drop = ones(4,4) .- (0.001f0 .* rand(4,4))
#        s = svd(drop)
#        drop = s.U*s.Vt |> nfae.machine
#        lift_mat = inv(cpu(drop))
#   end 
#   ctr = 1 
#   nfae.par = AE(:Par, nfae.par.in_dim,nfae.par.out_dim,nfae.par.widths,act_,nfae.machine)
#   nfae.state = AE(:State, nfae.state.in_dim,nfae.state.out_dim,nfae.state.widths,act_,nfae.machine)
#   pretrain(10000,batchsize,7e-2,ADAM(0.001))
#   
#
#   load_params(nfae)
#   drop = FileIO.load("/home/kaliam/NFAEdata/fluid-Hopf/drop.jld2","drop") |> nfae.machine
#   lift_mat = inv(cpu(drop))
#
#   train(nfae,nEpochs,batchsize, x_test, dx_test, alpha_test)
#   adamarg = 0.001
#   if ctr > nEpochs*(training_size/batchsize) 
#       break
#   end
#end
#
#save_params(nfae)
#FileIO.save("/home/kaliam/NFAEdata/fluid-Hopf/drop.jld2","drop",cpu(drop))
#FileIO.save("/home/kaliam/NFAEdata/fluid-Hopf/test_ind.jld2","test_ind",cpu(test_ind))
#
