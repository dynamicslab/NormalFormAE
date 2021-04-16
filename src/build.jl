mutable struct NFAE{x_name, z_name}
    model_x :: xModel{x_name}
    model_z :: NormalForm{z_name}
    data_dir :: Union{String,Nothing}
    exp_name :: Union{String,Nothing}
    training_size :: Int64
    test_size :: Int64
    training_data :: Union{Dict,Nothing}
    test_data :: Union{Dict,Nothing}
    state :: AE{:State} # Should be Flux.Chain
    par :: AE{:Par} # SHould be Flux.Chain
    trans :: Union{AE{:Trans},Nothing} # Should be Flux.Chain or nothing
    p_ae_state :: Float32
    p_ae_par :: Float32
    p_ae_trans :: Float32
    p_cons_x :: Float32
    p_cons_z :: Float32
    p_zero :: Float32
    p_orient :: Float32
    loss :: Any
    tscale :: Any
    machine :: Any # Flux.gpu/Flux.cpu
    nPlots :: Int64
    nEnsPlot :: Int64
    varPlot :: Float64
    ijulia :: Bool
    bigdata :: Bool
    # Plotting
    #live_plotter :: Plotter
    #nplots :: Int64 # How many plots from test data
    #nEns :: Float64 # How many in every ensemble 
    #varPlot :: Float64 # variance of initial values 
end
    
function NFAE(x_name, z_name, model_x, model_z,train_size, test_size,
              State :: AE{:State}, Par :: AE{:Par}, Trans,
              tscale_init,
              reg, machine,
              nPlots, nEnsPlot, varPlot, data_dir; ijulia = false, bigdata = false)
    # To get: training_data, test_data,NNs, loss function, plotter
    # data
    if data_dir == nothing
        training_data = Dict()
        test_data = Dict()
        training_data["x"], training_data["dx"], training_data["alpha"] =
            model_x.gen(train_size) 
        test_data["x"], test_data["dx"], test_data["alpha"] = model_x.gen(test_size) 
        # reg
        tscale = tscale_init |> machine
        println("Data and params unsaved, run save_data, save_params")
    else
        training_data = nothing
        test_data  = nothing
        println("Load datasets and parameters with load_data, load_params")
    end
    exp_name = join([String(x_name),String(z_name)],"-")
    p_ae_state = reg[1]
    p_ae_par = reg[2]
    p_ae_trans = reg[3]
    p_cons_x = reg[4]
    p_cons_z = reg[5]
    p_zero = reg[6]
    p_orient = reg[7]
    # SETUP PLOTTER!
    #live_plotter(x) = plotter(x)
    x_name = x_name
    z_name = z_name
    NFAE{x_name,z_name}(model_x,model_z,
                        data_dir, exp_name,
                        train_size,
                        test_size, training_data, test_data,
                        State, Par, Trans,
                        p_ae_state,p_ae_par, p_ae_trans, p_cons_x, p_cons_z, p_zero, p_orient,nothing,
                        tscale_init,machine,
                        nPlots, nEnsPlot, varPlot, ijulia, bigdata)
end

function Base.show(io::IO, nfae :: NFAE{x_name, z_name}) where {x_name, z_name}
    println("Normal Form Autoencoder Model:")
    println("System: ", String(x_name))
    println("Bifurcation: ", String(z_name))
    println("Data: ",nfae.training_size," (train), ",nfae.test_size, " (test)")
    if nfae.machine == gpu
        println("Ready to train on GPU")
    else
        println("Warning: Ready to train on CPU")
    end
end
