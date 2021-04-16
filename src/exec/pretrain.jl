function reg(enc,x)
    if typeof(enc).parameters[1] == :Par
        return Flux.mse(sign.(enc.decoder(enc.encoder(x))),sign.(x))
    else
        return 0.0f0
    end
end

function reg_cond(enc,x,norm)
    if typeof(enc).parameters[1] == :Par
        return reg(enc,x)*prod(size(x)) < 0.5f0 ? true : false
    else
        reg_func = Flux.mse(enc.decoder(enc.encoder(x)),x)
        return reg_func < norm ? true : false
    end
end


function pretrain_enc_state(enc, data, nit,norm, opt)
    ps = Flux.params(enc.encoder,enc.decoder)
    while !reg_cond(enc,data,norm) && nit >0
        res, back = Flux.pullback(ps) do
            state_enc_loss(enc,data)
        end # Performs autodiff step
        nit = nit - 1
        Flux.Optimise.update!(opt,ps,back(1f0))
        println(String(typeof(enc).parameters[1]),":",res)
    end
    return nit
end   
function pretrain_enc_par(enc, data, nit,norm, opt)
    ps = Flux.params(enc.encoder,enc.decoder)
    while !reg_cond(enc,data,norm) && nit >0
        res, back = Flux.pullback(ps) do
            par_enc(enc,data)
        end # Performs autodiff step
        nit = nit - 1
        Flux.Optimise.update!(opt,ps,back(1f0))
        println(String(typeof(enc).parameters[1]),":",res)
    end
    return nit
end   

function pretrain(nfae,nit,batchsize,norm,opt) 
    # Pretrain par
    old_nit = deepcopy(nit)
    par_train = Array(-1.0f0:0.01f0:1f0) |> nfae.machine
    par_train = reshape(par_train,1,length(par_train))
    while nit > 0
        nit = pretrain_enc(nfae.par, par_train, nit, norm,opt)
        if nit > 0
            nit = -1
        else
            nfae.par = AE(:Par, nfae.par.in_dim,nfae.par.out_dim,nfae.par.widths,Symbol(nfae.par.act),nfae.machine) # respawn
            nit = 500
        end    
    end
    println("Paramater AE pretrained.")
    # Pretrain state
    state_train, dx, alpha = makebatch(nfae.training_data,batchsize,2)
    state_train = state_train |> nfae.machine
    dx = nothing
    alpha = nothing
    nit = old_nit
    while nit > 0
        nit = pretrain_enc(nfae.state, state_train, nit,norm,opt)
        if nit > 0
            nit = -1
        else
            nfae.state = AE(:State, nfae.state.in_dim,nfae.state.out_dim,nfae.state.widths,Symbol(nfae.state.act),nfae.machine) # respawn
            nit = 500
        end    
    end
    println("State AE pretrained.")	
end
