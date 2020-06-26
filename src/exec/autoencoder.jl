
function get_activation(x::String)
    if x == "relu"
        activation_ = relu
    elseif x == "elu"
        activation_ = elu
    elseif x == "sigmoid"
        activation_ = sigmoid
    elseif x == "leakyrelu"   
        activation_ = leakyrelu
    elseif x == "tanh"
        activation_ = Flux.tanh
    elseif x == "id"
        activation_ = -1
    end

    return activation_
end

function act_der(act::String,x,avgder=0)
    if act == "sigmoid"
        der_ = sigmoid.(x) .*(1.0f0 .- sigmoid.(x))
    elseif act == "tanh"
        der_ = 1.0f0 .- (tanh.(x)).^2
    elseif act == "relu"
        if avgder == 1
            der_ = 0.5f0 .*(sign.(x) .+ 1.0f0)
        else
            der_ = 0.5f0 .*(sign.(x) .+ abs.(sign.(x)))
        end
    elseif act == "leakyrelu"
        if avgder == 1
            der_ = 0.5f0 .*(sign.(x) .+ 1.0f0) .+ 0.01f0./2.0f0 .*( -sign.(x) .+ 1.0f0)
        else
            der_ = 0.5f0 .*(sign.(x) .+ abs.(sign.(x))) .- 0.01f0./2.0f0 .*( sign.(x) .- abs.(sign.(x)))
        end
    elseif act == "elu"
        if avgder == 1
            der_ = 0.5f0 .*(sign.(x) .+ 1.0f0) .+ exp.(x)./2.0f0 .*( -sign.(x) .+ 1.0f0)
        else
            der_ = 0.5f0 .*(sign.(x) .+ abs.(sign.(x))) .- exp.(x)./2.0f0 .*( sign.(x) .- abs.(sign.(x)))
        end
    end
    return der_
end
        
function get_autoencoder(args::Dict)
    function get_NN_Flux(widths::Array,act_widths::Array)
        local NN = []
        for i in 1:(size(widths)[1]-1)
            activation_ = get_activation(act_widths[i])
            if activation_ == -1
                append!(NN,[Dense(widths[i],widths[i+1])])
            else
                append!(NN,[Dense(widths[i],widths[i+1],activation_)])
            end
        end
        return Chain(NN...)
    end
    
    AE_widths = args["AE_widths"]
    Par_widths = args["Par_widths"]
    AE_acts = args["AE_acts"]
    Par_acts = args["Par_acts"]

    encoder = get_NN_Flux(AE_widths,AE_acts)
    decoder = get_NN_Flux(reverse(AE_widths),[reverse(args["AE_acts"])[2:end];"id"])
    par_encoder = get_NN_Flux(Par_widths,Par_acts)
    par_decoder = get_NN_Flux(reverse(Par_widths),[reverse(args["Par_acts"])[2:end];"id"])
    u0_train = rand(args["z_dim"])

    return encoder, decoder, par_encoder,par_decoder, u0_train
end

function dt_NN(NN, input_, left_dt, acts)
    # Given input, left_dt and neural net NN, compute right hand side time-derivative via chain rule
    # As per Champion et al. (PNAS 2019)
    W_,b_ = Flux.params(NN.layers[1])
    l = W_*input_.+b_
    dl = W_*left_dt
    act = acts[1]
    nlayers = div(length(Flux.params(NN)),2)
    for i=2:nlayers
        W_,b_ = Flux.params(NN.layers[i])
        act_fun = get_activation(act)
        if act != "id"
            dl = W_*(act_der(act,l).*dl)
            l = W_*act_fun.(l) .+ b_
        else
            dl = W_*dl
            l = W_*l .+ b_
        end
        act = acts[i]
    end
    act_fun = get_activation(act)
    if act != "id"
        dl = act_der(act,l).*dl
        l = act_fun.(l)
    end     
    return dl
end

function build_loss(args,normalform_,nf_solve,encoder, decoder, par_encoder,par_decoder, u0_train)
    t_batch = range(0.0f0,Float32(args["tspan"][1]),length = args["tsize"])
    ode_prob_temp = ODEProblem(nf_solve,u0_train,(0.0f0,Float32(args["tspan"][2])),gpu(rand(args["par_dim"])))
    solve(ode_prob_temp)
    ode_prob = x -> remake(ode_prob_temp,p=x)
    solve(ode_prob(gpu(rand(1))))
    #ODE solve to be used for training
    function predict_ODE_solve(x)
        return Array(solve(ode_prob(x),Tsit5(),saveat=t_batch,reltol=1e-4)) 
    end
    
    function loss_(in_,dx_,par_)
        enc_ = encoder(in_)
        dz1 = dt_NN(encoder,in_,dx_,args["AE_acts"])
        enc_par = par_encoder(par_)
        dec_par = par_decoder(enc_par)
        dec_ = decoder(enc_)
        #println(size(enc_par))
        enc_ODE_solve = predict_ODE_solve(enc_par[:,1])
        dx1 = dt_NN(decoder,enc_,normalform_(enc_,enc_par),[reverse(args["AE_acts"])[2:end];"id"])
        loss_datafid = args["P_DataFid"]*Flux.mse(in_,dec_)
        loss_dx = args["P_dx"]*Flux.mse(dx_,dx1)
        loss_dz = args["P_dz"]*Flux.mse(dz1,normalform_(enc_,enc_par))
        loss_cons = args["P_cons"]*Flux.mse(enc_ODE_solve,enc_)
        loss_par = args["P_par"]*Flux.mse(dec_par,par_)
        loss_dec2 = args["P_dec2"]*Flux.mse(in_,decoder(enc_ODE_solve))
        args["loss_AE"] = loss_datafid
        args["loss_dxdt"] = loss_dx
        args["loss_dzdt"] = loss_dz
        args["loss_par"] = loss_par
        args["loss_cons"] = loss_cons
        loss_total = loss_datafid  + loss_dz + loss_dx + loss_par + loss_cons + loss_dec2
        args["loss_total"] = loss_total
        return loss_total
    end
    return loss_
end

        
        
        
    
        
