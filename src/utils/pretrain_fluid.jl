
if nfae.state.act == "id"
    act_ = nothing
else
    act_ = Symbol(nfae.state.act)
end

sqnorm(x) = sum(abs2,x)

function state_enc_loss(data)
    return Flux.mse(nfae.state.decoder(nfae.state.encoder(data)),data)  
end

function par_enc_loss(data)
    return Flux.mse(nfae.par.decoder(nfae.par.encoder(data)),data) + Flux.mse(sign.(nfae.par.encoder(data)),sign.(data))
end

function cond_state(data,norm)
    return state_enc_loss(data) < norm ? true : false
end

function cond_par(data)
    return Flux.mse(sign.(nfae.par.encoder(data)),sign.(data)) > 0.0f0 ? false : true
end
    
function pretrain_enc_state(data, nit,norm, opt)
    ps = Flux.params(nfae.state.encoder,nfae.state.decoder)
    while !cond_state(data,norm) && nit >0
        res, back = Flux.pullback(ps) do
            state_enc_loss(data) 
        end # Performs autodiff step
        nit = nit - 1
        Flux.Optimise.update!(opt,ps,back(1f0))
        println("State",":",state_enc_loss(data)," res:", res)
    end
    return nit
end   
function pretrain_enc_par(data, nit,opt)
    ps = Flux.params(nfae.par.encoder,nfae.par.decoder)
    while !cond_par(data) && nit >0
        res, back = Flux.pullback(ps) do
            #println(par_enc_loss(data))
            par_enc_loss(data)
        end # Performs autodiff step
        nit = nit - 1
        Flux.Optimise.update!(opt,ps,back(1f0))
        # println("Par:",par_enc_loss(data)," res:", res)
    end
    return nit
end   

function pretrain(nit,batchsize,norm,opt) 
    # Pretrain par
    old_nit = deepcopy(nit)
    par_train = Array(-1.0f0:0.01f0:1f0) |> nfae.machine
    par_train = reshape(par_train,1,length(par_train))
    nit = 500
    while nit > 0
        nit = pretrain_enc_par(par_train, nit,opt)
        if nit > 0
            nit = -1
        else
            nfae.par = AE(:Par, nfae.par.in_dim,nfae.par.out_dim,nfae.par.widths,act_,nfae.machine) # respawn
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
        nit = pretrain_enc_state(drop*state_train, nit,norm,opt)
        if nit > 0
            nit = -1
        else
            nfae.state = AE(:State, nfae.state.in_dim,nfae.state.out_dim,nfae.state.widths,act_,nfae.machine) # respawn
            nit = old_nit
        end    
    end
    println("State AE pretrained.")	
end
