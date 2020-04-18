using Flux

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
        activation_ = tanh
    elseif x == "id"
        activation_ = -1
    end

    return activation_
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
    Hom_widths = args["Hom_widths"]
    AE_acts = args["AE_acts"]
    Hom_acts = args["Hom_acts"]

    encoder = get_NN_Flux(AE_widths,AE_acts)
    decoder = get_NN_Flux(reverse(AE_widths),reverse(AE_acts))
    hom_encoder = get_NN_Flux(Hom_widths,Hom_acts)
    hom_decoder = get_NN_Flux(reverse(Hom_widths),reverse(Hom_acts))

    return encoder, decoder, hom_encoder, hom_decoder
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
        if act == "elu"
            dl = W_*(min.(exp.(l)).*dl)
        elseif act == "relu"
            dl = W_*(Flux.TrackedArray(Float32.(l.>0)).*dl)
        elseif act == "sigmoid"
            dl = W_*((1.0 .- sigmoid.(l)).*sigmoid.(l).*dl)
        elseif act == "id"
            dl = W_*dl
        end
        if get_activation(act) == -1
            l = W_*l .+ b_
        else
            l = W_*get_activation(act).(l).+b_
        end
        act = acts[i]
    end
    return dl
end


function build_loss(args,normalform_,encoder, decoder, hom_encoder, hom_decoder)
    function loss_(in_,dx_)
        enc_ = encoder(in_)
        hom_ = hom_encoder(enc_)
        dz1 = dt_NN(encoder,in_,dx_,args["AE_acts"])
        dz_ = dt_NN(hom_encoder,enc_,dz1,args["Hom_acts"])
        hom_dec_ = hom_decoder(hom_)
        dec_ = decoder(hom_dec_)
        dx1 = dt_NN(hom_decoder,hom_,normalform_(hom_),reverse(args["Hom_acts"]))
        dx_predict = dt_NN(decoder,hom_dec_,dx1,reverse(args["AE_acts"]))
        loss_datafid = args["P_DataFid"]*Flux.mse(in_,dec_)
        loss_hom = args["P_Hom"]*Flux.mse(hom_,hom_dec_)
        loss_dx = args["P_dx"]*Flux.mse(dx_,dx_predict)
        loss_dz = args["P_dz"]*Flux.mse(dz_,normalform_(hom_))
        args["loss_AE"] = loss_datafid
        args["loss_Hom"] = loss_hom
        args["loss_dxdt"] = loss_dx
        args["loss_dzdt"] = loss_dz
        loss_total = loss_datafid + loss_hom + loss_dx + loss_dz
        args["loss_total"] = loss_total
        return loss_total
    end
    return loss_
end

        
        
        
    
        
