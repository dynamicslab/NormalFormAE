using Pkg
Pkg.activate(".")
using DiffEqFlux, DiffEqSensitivity, Flux, OrdinaryDiffEq, Zygote, Test, DifferentialEquations

# Training hyperparameters
nBatchsize = 10
nEpochs = 10
tspan = 5.0
tsize = 100
nbatches = div(tsize,nBatchsize)

# ODE solve
function lotka_volterra(du,u,p,t)
    dx = Zygote.Buffer(u,size(u)[1])
    dx[1] = u[1]*(p[1]-p[2]*u[2])
    dx[2] = u[2]*(p[3]*u[1]-p[4])
    du .= copy(dx)
    nothing
end

# Define parameters and initial conditions for data
p = Float32[2.2, 1.0, 2.0, 0.4] 
u0 = Float32[0.01, 0.01]

t = range(0.0,tspan,length=tsize)

# Define ODE problem and generate data
prob = ODEProblem(lotka_volterra,u0,(0.0,tspan),p)
yy = Array(solve(prob,saveat=t))
y_original = Array(solve(prob,saveat=t))
yy = yy .+ yy*(0.01.*rand(size(yy)[2],size(yy)[2])) # Creates noisy, translated data

data = Float32.(yy) |> gpu

# Define autoencoder networks
NN_encode = Chain(Dense(2,10),Dense(10,10),Dense(10,2)) |> gpu
NN_decode = Chain(Dense(2,10),Dense(10,10),Dense(10,2)) |> gpu

# Define new ODE problem for "batch" evolution
t_batch = range(0.0f0,Float32(tspan/nbatches),length = nBatchsize)
prob2 = ODEProblem(lotka_volterra,u0,(0.0f0,Float32(tspan/nbatches)),p)

# ODE solve to be used for training
function predict_ODE_solve(x)
    return Array(solve(prob2,Tsit5(),u0=x,saveat=t_batch,reltol=1e-4)) 
end

function loss_func(data_)
    enc_ = NN_encode(data_)
    # Solve ODE using initial values from multiple points in enc_.
    # Note: reduce(hcat,[..]) gives a mutating arrays error
    enc_ODE_solve = hcat([predict_ODE_solve(enc_[:,(i-1)*nBatchsize+1]) for i in 1:nbatches]...) |> gpu
    dec_1 = NN_decode(enc_ODE_solve)
    dec_2 = NN_decode(enc_)
    loss = Flux.mse(data_,dec_1) + Flux.mse(data_,dec_2) + 0.001*Flux.mse(enc_,enc_ODE_solve)
    args_["loss"] = loss
    return loss
end

opt = ADAM(0.001)

loss_func(data) # This works

for ep in 1:nEpochs
    global args_
    @info "Epoch $ep"
    Flux.train!(loss_func, Flux.params(NN_encode,NN_decode), [(data)], opt)
    loss_ = args_["loss"]
    println("loss: $(loss_)")
end
