function save_posttrain(nfae::NFAE)
    dir_ = nfae.data_dir
    exp_  = nfae.exp_name
    weights_encoder = Flux.params(cpu(nfae.state.encoder))
    weights_decoder = Flux.params(cpu(nfae.state.decoder))
    weights_par_encoder = Flux.params(cpu(nfae.par.encoder))
    weights_par_decoder = Flux.params(cpu(nfae.par.decoder))
    if nfae.trans != nothing
        weights_trans_encoder = Flux.params(cpu(nfae.trans.encoder))
        weights_trans_decoder = Flux.params(cpu(nfae.trans.decoder))
    end
    tscale = cpu(nfae.tscale)
    path_ = pwd()
    if dir_ ∉ readdir(joinpath(path_,".."))
        mkpath(joinpath(path_,"..",dir_))
        mkpath(joinpath(path_,"..",dir_,exp_))
    elseif exp_ ∉ readdir(joinpath(path_,"..",dir_))
        mkpath(joinpath(path_,"..",dir_,exp_))
    end
    path_data = joinpath(path_,"..",dir_,exp_)

    training_data = nfae.training_data
    test_data = nfae.test_data
    for (key,item) in training_data
        training_data[key] = cpu(training_data[key])
    end
    for (key,item) in test_data
        test_data[key] = cpu(test_data[key])
    end
    
    BSON.@save "$(path_data)/encoder.bson" weights_encoder
    BSON.@save "$(path_data)/decoder.bson" weights_decoder
    BSON.@save "$(path_data)/par_encoder.bson" weights_par_encoder
    BSON.@save "$(path_data)/par_decoder.bson" weights_par_decoder
    if nfae.trans != nothing
        BSON.@save "$(path_data)/trans_encoder.bson" weights_trans_encoder
        BSON.@save "$(path_data)/trans_decoder.bson" weights_trans_decoder
    end
    FileIO.save("$(path_data)/training_data.jld2","training_data",training_data)
    FileIO.save("$(path_data)/test_data.jld2","test_data",test_data)
    FileIO.save("$(path_data)/tscale.jld2","tscale",tscale)
end

function load_posttrain(nfae::NFAE)
    
    dir_ = nfae.data_dir
    exp_  = nfae.exp_name
    # Load autoencoders

    path_ = pwd()
    path_ = joinpath(path_,"..",dir_,exp_)
    BSON.@load "$(path_)/encoder.bson" weights_encoder
    BSON.@load "$(path_)/decoder.bson" weights_decoder
    BSON.@load "$(path_)/par_encoder.bson" weights_par_encoder
    BSON.@load "$(path_)/par_decoder.bson" weights_par_decoder

    if nfae.trans != nothing
        BSON.@load "$(path_)/trans_encoder.bson" weights_trans_encoder
        BSON.@load "$(path_)/trans_decoder.bson" weights_trans_decoder  
    end
    
    # Load train/test data
    Flux.loadparams!(nfae.state.encoder,weights_encoder)
    Flux.loadparams!(nfae.state.decoder,weights_decoder)
    Flux.loadparams!(nfae.par.encoder,weights_par_encoder)
    Flux.loadparams!(nfae.par.decoder,weights_par_decoder)
    if nfae.trans != nothing
        Flux.loadparams!(nfae.trans.encoder,weights_trans_encoder)
        Flux.loadparams!(nfae.trans.decoder,weights_trans_decoder)
    end
    
    try
        tscale = FileIO.load("$(path_)/tscale.jld2","tscale")
        nfae.tscale = tscale |> nfae.machine
    catch e
        println("tscale wasn't saved")
    end
    nfae.training_data = FileIO.load("$(path_)/training_data.jld2","training_data")
    nfae.test_data = FileIO.load("$(path_)/test_data.jld2","test_data")
    for (key,_) in nfae.training_data
        nfae.training_data[key] = nfae.machine(training_data[key])
    end
    for (key,_) in nfae.test_data
        nfae.test_data[key] = nfae.machine(test_data[key])
    end
    nothing
end

    
