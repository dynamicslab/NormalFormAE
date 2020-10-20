function save_posttrain(args,NN,training_data, test_data)
    dir_ = args["DataDir"]
    exp_  = args["ExpName"]
    weights_encoder = params(cpu(NN["encoder"]))
    weights_decoder = params(cpu(NN["decoder"]))
    weights_par_encoder = params(cpu(NN["par_encoder"]))
    weights_par_decoder = params(cpu(NN["par_decoder"]))
    tscale = cpu(NN["tscale"])
    BSON.@save "$(dir_)/Final/$(exp_)/encoder.bson" weights_encoder
    BSON.@save "$(dir_)/Final/$(exp_)/decoder.bson" weights_decoder
    BSON.@save "$(dir_)/Final/$(exp_)/par_encoder.bson" weights_par_encoder
    BSON.@save "$(dir_)/Final/$(exp_)/par_decoder.bson" weights_par_decoder
    FileIO.save("$(dir_)/Final/$(exp_)/training_data.jld2","training_data",training_data)
    FileI0.save("$(dir_)/Final/$(exp_)/test_data.jld2","test_data",test_data)
    FileIO.save("$(dir_)/Final/$(exp_)/tscale.jld2","tscale",tscale)
    FileIO.save("$(dir_)/Final/$(exp_)/args.jld2","args",args)
end

function load_posttrain(args)
    NN = gen_NN(args)
    dir_ = args["DataDir"]
    exp_  = args["ExpName"]
    # Load autoencoders
    
    BSON.@load "$(dir_)/Final/$(exp_)/encoder.bson" weights_encoder
    BSON.@load "$(dir_)/Final/$(exp_)/decoder.bson" weights_decoder
    BSON.@load "$(dir_)/Final/$(exp_)/par_encoder.bson" weights_par_encoder
    BSON.@load "$(dir_)/Final/$(exp_)/par_decoder.bson" weights_par_decoder

    # Load train/test data
    Flux.loadparams!(NN["encoder"],weights_encoder)
    Flux.loadparams!(NN["decoder"],weights_decoder)
    Flux.loadparams!(NN["par_encoder"],weights_par_encoder)
    Flux.loadparams!(NN["par_decoder"],weights_par_decoder)

    try
        tscale = FileIO.load("$(dir_)/Final/$(exp_)/tscale.jld2","tscale")
        NN["tscale"] = tscale |> gpu
    catch e
        println("tscale wasn't saved")
    end
    training_data = FileIO.load("$(dir_)/Final/$(exp_)/training_data.jld2","training_data")
    test_data = FileIO.load("$(dir_)/Final/$(exp_)/test_data.jld2","test_data")
    try
        args = FileI0.load("$(dir_)/Final/$(exp_)/args.jld2","args")
    catch e
        println("args were not saved")
    end

    return args, NN, training_data, test_data
end

    
