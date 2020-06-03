using Pkg
Pkg.activate(".")
using DiffEqFlux, DiffEqSensitivity, Flux, OrdinaryDiffEq, Zygote, Test, DifferentialEquations #using Plots
using Base.Iterators: partition


nBatchsize = 50
nEpochs = 1000
tspan = 5.0
tsize = 100
nbatches = div(tsize,nBatchsize)


# function lotka_volterra(du,u,p,t)
#     dx = Zygote.Buffer(u,size(u)[1])
#     dx[1] = u[1]*(p[1]-p[2]*u[2])
#     dx[2] = u[2]*(p[3]*u[1]-p[4])
#     du .= copy(dx)
#     # du .= [u[1]*(p[1]-p[2]*u[2]), u[2]*(p[3]*u[1]-p[4])]
#     nothing
# end



function lotka_volterra(u,p,t)
    x, y = u
    α, β, δ, γ = p
    du = [(α - β*y)x,
    (δ*x - γ)y]
    return du
end

function lotka_volterra_train(du,u,p,t)
    du .= lotka_volterra(u,p,t)
end

p = Float32[2.2, 1.0, 2.0, 0.4] 
u0 = Float32[0.01, 0.01]

t = range(0.0,tspan,length=tsize)

prob = ODEProblem(lotka_volterra,u0,(0.0,tspan),p)
yy = Array(solve(prob,saveat=t))
y_original = Array(solve(prob,saveat=t))
for i in 1:1
    global yy
    yy = yy .+ yy*(0.01.*rand(size(yy)[2],size(yy)[2]))
end

#data = [(yy[:,i]) for i in partition(1:size(yy)[2],nBatchsize)]
data = Float32.(yy) 
init_ = hcat([data[:,(i-1)*nBatchsize+1] for i in 1:nbatches]...) 

NN1 = Chain(Dense(2,10),Dense(10,10),Dense(10,2)) 
NN2 = Chain(Dense(2,10),Dense(10,10),Dense(10,2)) 

t_batch = Float32.((0.0,tspan/nbatches))
t_batch1 = range(0.0f0,Float32(tspan/nbatches),length = nBatchsize)

args_ = Dict()
# u0 = gpu(u0)
# p = gpu(p)

prob2 = ODEProblem(lotka_volterra_train,u0,(0.0f0,Float32(tspan/nbatches)),p)
#u0 = hcat([u0 for i in 1:nbatches]...)
#u0 = gpu(u0)

function predict_rd(x)
    return Array(solve(prob2,Tsit5(),u0=x,saveat=t_batch1,reltol=1e-4))
end

function gen_loss(dat_)
    y1 = NN1(dat_)
    y2 = hcat([predict_rd(y1[:,(i-1)*nBatchsize+1]) for i in 1:nbatches]...)
    # y3_temp = Zygote.Buffer(y1,size(y1)[1],size(y1)[2])
    # for i in 1:nbatches
    #     y3_temp[:,(i-1)*nBatchsize+1:i*nBatchsize] = y2[:,i,:]
    # end
    # y3 = copy(y3_temp)
    y4 = NN2(y2)
    y5 = NN2(y1)
    loss = Flux.mse(dat_,y4) + Flux.mse(dat_,y5) + 0.001*Flux.mse(y1,y2)
    args_["loss"] = loss
    return loss
end


# function predict_rd(x)
#     println(size(x))
#     y1 = NN1(x)
#     #prob_ = ODEProblem(lotka_volterra,u0,(0.0,50.0),p)
#     y2 = Array(concrete_solve(prob2,Tsit5(),saveat=t_batch1,reltol=1e-4)) |> gpu
#     println(size(y2))
#     y3 = NN2(y2)
#     loss = Flux.mse(y3,x) + Flux.mse(y1,y2)
#     args["loss"] = loss
#     return loss
# end
# loss_rd() = sum(abs2,x-1 for x in predict_rd())

opt = ADAM(0.001)

gen_loss(data)

#dat_ = [(data)] |> gpu

# train_steps = 0
for ep in 1:nEpochs
    global args_
    @info "Epoch $ep"
    Flux.train!(gen_loss, Flux.params(NN1,NN2), [(data)], opt)
    loss_ = args_["loss"]
    println("loss: $(loss_)")
end

# Display the ODE with the current parameter values.
# Flux.train!(predict_rd, Flux.params(u0,NN1,NN2), Iterators.repeated((), 100), opt, cb = cb)
