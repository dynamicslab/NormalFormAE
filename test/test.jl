using Pkg
Pkg.activate(".")
using LinearAlgebra, Flux, Zygote
using Plots, LaTeXStrings, Printf, Measures
using FileIO, BSON
include("src/ae.jl")
include("src/exec/backpass_explicit.jl")
include("problems/lp.jl")
include("src/model.jl")
include("src/build.jl")
include("src/exec/build_loss.jl")
include("src/exec/plot.jl")
include("src/exec/post_train.jl")


# Test 1
state = nothing
par=  nothing
trans = nothing
try
    global state = AE(:State, 1,1,[16,16],:elu,gpu)
    global par = AE(:Par, 1,1,[16,16],:elu,gpu)
    global trans = AE(:Trans, 1,1,[16,16],:elu,gpu)
    println("AE construction passed.")
catch e
    printn("AE construction failed.")
end

# Test 2
model_x = nothing
model_z = nothing
try
    global model_x = xModel(:Scalar, 1, 1, 100,[0.0,0.05],0.1,0.1,[0.0],[0.0],dxdt_rhs,dxdt_solve,Dict())
    global model_z = NormalForm(:LP, 1,1, dzdt_rhs, dzdt_solve)
    println("Model construction passed.")
catch e
    println("Model construction failed.")
end

# Test 3
nfae = nothing
try
    data_dir = "NormalFormAEData"
    global nfae = NFAE(:Scalar, :LP, model_x, model_z, 100, 10, state, par, trans, [10f0],
                       ones(Float32,7),gpu, 2,2,0.01,"NormalFormAEData")
    load_posttrain(nfae)
    println("Models, AE combined, data constructed.")
catch e
    println("Combination + data construction failed.")
end

# Test 4
test_x = nothing
test_dx = nothing
test_a = nothing
loss = nothing
try
    global test_x = reduce(hcat,[nfae.test_data["x"][:,:,i] for i in 1:size(nfae.test_data["x"])[end]])
    global test_dx = reduce(hcat,[nfae.test_data["dx"][:,:,i] for i in 1:size(nfae.test_data["dx"])[end]])
    global test_a = nfae.test_data["alpha"]
    global loss = nfae(test_x,test_dx,test_a)
    if typeof(loss) <: Float32
        println("Loss output is fine.")
    else
        println("Loss is okay, but output is weird.")
    end
catch e
    println("Some error in the loss function.")
end

# Test 5
try
    ps = Flux.params(nfae.state.encoder, nfae.state.decoder, nfae.par.encoder, nfae.par.decoder, nfae.trans.encoder, nfae.trans.decoder)
    loss_, back_ = Flux.pullback(ps) do
        nfae(test_x, test_dx, test_a)
    end
    println("Pullback is okay.")
    Flux.Optimise.update!(ADAM(0.01),ps,back_(1f0))
    println("Training successful.")
catch e
    println("Problem in training.")
end

# Test 6    
try
    p = gen_plot(nfae.model_z.z_dim, nfae.nPlots)
    println("Plot generated.")
    plotter(nfae,1,p,cpu(test_x),cpu(test_a),0.1,0.1)
    println("Plot done.")
    plot(p...)
    println("Plot shown.")
catch e
    println("Something wrong with plotting.")
end

# Test 7
try
    nfae.data_dir = "NormalFormAEData"
    save_posttrain(nfae)
    println("Saving success.")
catch e
    println("Problem with saving.")
end
