using Pkg
Pkg.activate("/home/manu/Documents/Work/NormalFormAE/.")
using NormalFormAE, Flux, Zygote, Plots, DifferentialEquations
include("/home/manu/Documents/Work/NormalFormAE/problems/nf.jl")
include("/home/manu/Documents/Work/NormalFormAE/run/run_nf_old.jl")

x_model_name = :nf
z_model_name = :Hopf
x_dim = 128
par_dim = 1
z_dim = 2
tsize = 500
tspan = [0.0,200.0]
xVar = 0.1
aVar = 0.5
mean_ic_x = [0.0] 
mean_ic_a = [0.0]
x_rhs = dxdt_rhs
x_solve = dxdt_solve
machine = gpu
 

model_x = xModel(x_model_name, x_dim, par_dim, tsize,tspan,xVar,aVar,mean_ic_x,mean_ic_a,x_rhs,x_solve, args)
model_z = NormalForm(:Hopf,z_dim ,par_dim, dzdt_rhs, dzdt_solve)

state = AE(:State, 128,2, [64,32,16],:elu,machine)
par = AE(:Par, 1,1,[16,16],:elu,machine)
trans = AE(:Trans,1,2,[16,16],:elu, machine)


tscale_init = [1f0] |> machine

training_size = 1000
test_size = 20


# data_dir = nothing

P_reg = [1.0f0, 1.0f0, 1.0f0, 0.001f0, 0.001f0, 0.001f0, 1.0f0]

data_dir =  "NormalFormAEData_old"

nfae = NFAE(:nf, :Hopf, model_x, model_z, training_size, test_size, state, par, nothing, tscale_init,
                       P_reg,machine, 10,20,0.1,data_dir)

# ---------------------------------- Training-----------------------------------------------------------
load_posttrain(nfae)

nfae.par =  AE(:Par, 1,1,[16,16],:elu,machine)

while sum(sign.(cpu(nfae.par.encoder)(nfae.test_data["alpha"])) .- sign.(nfae.test_data["alpha"])) > 2.0
    state_new = AE(:State, x_dim,z_dim, [64,32,16],:elu,machine)
    par_new = AE(:Par, 1,1,[16,16],:elu,machine)
    # trans = AE(:Trans,1,2,[16,16],:elu,machine)
    nfae.state = state_new
    nfae.par = par_new
    # nfae.trans = trans
    nfae.tscale = [0.01f0] |> machine
    println("changed")
end
nfae.tscale = [1f0] |> machine

save_posttrain(nfae)

nEpochs = 200
batchsize = 100
ctr = 0
p = gen_plot(nfae.model_z.z_dim, nfae.nPlots)

# Load test data
x_test = reduce(hcat,[nfae.test_data["x"][:,:,i] for i in 1:nfae.test_size]) |> nfae.machine
dx_test = reduce(hcat,[nfae.test_data["dx"][:,:,i] for i in 1:nfae.test_size]) |> nfae.machine
alpha_test = nfae.test_data["alpha"] |> nfae.machine
# x_test = nfae.test_data["x"] |> nfae.machine
# dx_test = nfae.test_data["dx"] |> nfae.machine
# alpha_test = nfae.test_data["alpha"] |> nfae.machine

# nfae.training_data["dx"] = nfae.training_data["dxdt"];

# Train!
for i in 1:nEpochs
    for j in 1:div(training_size, batchsize)
        global ctr
        x_, dx_, alpha_ = makebatch(nfae.training_data, batchsize,j) |> nfae.machine # batcher
        ps = Flux.params(nfae.state.encoder, nfae.state.decoder,
                         nfae.par.encoder, nfae.par.decoder) # defines NNs to be trained
        res, back = Flux.pullback(ps) do
            nfae(x_,dx_,alpha_)
        end # Performs autodiff step
        Flux.Optimise.update!(ADAM(0.001),ps,back(1f0)) # Updates training parameters
        loss_train = res
        loss_test = nfae(x_test,dx_test,alpha_test)
        println("Epoch: ",i, " Batch: ",j, " Train loss: ", loss_train, " Test loss: ", loss_test)
        plotter(nfae,ctr,p,cpu(x_test),cpu(alpha_test),loss_train,loss_test)
        ctr = ctr + 1
    end
    save_posttrain(nfae)
end


