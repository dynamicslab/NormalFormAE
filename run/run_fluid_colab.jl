using Pkg
path = joinpath(pwd(),"NormalFormAE")
Pkg.activate(joinpath(path,"."))
Pkg.instantiate()
# ENV["JULIA_CUDA_VERBOSE"] = true
# ENV["JULIA_CUDA_MEMORY_POOL"] = "split" # Efficient allocation to GPU (Julia garbage collection is inefficient for this code apparently)
# ENV["JULIA_CUDA_MEMORY_LIMIT"] = 8000_000_000
using NormalFormAE, Flux, Zygote, Plots, DifferentialEquations
include(joinpath(path,"problems/nf.jl"))
include(joinpath(path,"run/run_nf_old.jl"))

# This is the experiment for the fluid flow model. Data was obtained by simulating the fluid flow past
# a cylinder example in ViscousFlows.jl. Time derivatives are obtained numerically from data, using
# first order finite differences. The spatial domain is 78 x 42, which implies 3276 grid points.
# The bifurcation (we estimate from data, to make this exampple data-driven) occurs at Re = 53.29317.

# dt = 1.2
# Scaling for alpha = 28

x_model_name = :fluid
z_model_name = :Hopf
x_dim = 3276
par_dim = 1
z_dim = 2
tsize = 250
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

state = AE(:State, x_dim,z_dim, [400,50,16],:elu,machine)
par = AE(:Par, 1,1,[16,16],:elu,machine)
trans = AE(:Trans,1,2,[16,16],:elu,machine)

tscale_init = [0.01f0] |> machine

training_size = 230
test_size = 20

data_dir = "content/drive/My Drive/Work/NormalFormAEData"

P_reg = [1.0f0, 1.0f0, 1.0f0, 0.001f0, 0.001f0, 1.0f0, 1.0f0]

nfae = NFAE(:fluid, :Hopf, model_x, model_z, training_size, test_size, state, par, trans, tscale_init,
            P_reg,machine, 10,20,0.1,data_dir)

load_posttrain(nfae)

state = AE(:State, x_dim,z_dim, [400,50,16],:elu,machine)
par = AE(:Par, 1,1,[16,16],:elu,machine)
trans = AE(:Trans,1,2,[16,16],:elu,machine)
nfae.state = state
nfae.par = par
nfae.trans = trans
nfae.tscale = [0.1f0] |> machine


# ---------------------------------- Training-----------------------------------------------------------

nEpochs = 100
batchsize = 23
ctr = 0
p = gen_plot(nfae.model_z.z_dim, nfae.nPlots)
for i in 1:nEpochs
    for j in 1:div(training_size, batchsize)
        global ctr
        x_, dx_, alpha_ = makebatch(nfae.training_data, batchsize,j) |> nfae.machine # batcher
        ps = Flux.params(nfae.state.encoder, nfae.state.decoder,
                         nfae.par.encoder, nfae.par.decoder,
                         nfae.trans.encoder, nfae.trans.decoder,
                         nfae.tscale) # defines NNs to be trained
        res, back = Flux.pullback(ps) do
            nfae(x_,dx_,alpha_)
        end # Performs autodiff step
        Flux.Optimise.update!(ADAM(0.001),ps,back(1f0)) # Updates training parameters
        loss_train = res
        x_ = reduce(hcat,[nfae.test_data["x"][:,:,i] for i in 1:nfae.test_size]) |> nfae.machine
        dx_ = reduce(hcat,[nfae.test_data["dx"][:,:,i] for i in 1:nfae.test_size]) |> nfae.machine
        alpha_ = reduce(hcat,[nfae.test_data["alpha"][:,i] for i in 1:nfae.test_size]) |> nfae.machine
        loss_test = nfae(x_,dx_,alpha_)
        println("Epoch: ",i, " Batch: ",j, " Train loss: ", loss_train, " Test loss: ", loss_test)
        plotter(nfae,ctr,p,cpu(x_),cpu(alpha_),loss_train,loss_test)
        ctr = ctr + 1
    end
    save_posttrain(nfae)
end
        