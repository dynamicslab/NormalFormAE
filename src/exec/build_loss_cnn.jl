function (nfae::NFAE{xname,zname})(in_, dx_, par_) where {xname,zname}
    
    bsize = div(size(in_)[end], nfae.model_x.tsize)
    tdim = nfae.model_x.tsize
    n_ = nfae.model_x.x_dim
    p_ = nfae.model_z.par_dim
    
    state_enc = nfae.state.encoder(in_)    

    par_enc = nfae.par.encoder(par_)
    id_ = repeat(Matrix{Float32}(I,bsize,bsize),inner=(1,tdim-1)) |> nfae.machine
    
    par_dec = nfae.par.decoder(par_enc)
    state_dec = nfae.state.decoder(state_enc)

    # Consistency losses
    #  dz_ = dt_NN(nfae.state.encoder,in_,dx_,nfae.state.act)
    dz_ = reduce(hcat,[diff(state_enc[:,(k-1)*tdim+1:k*tdim]./Float32[(nfae.model_x.tspan[end]/(tdim))],dims=2) for k in 1:bsize]) |> nfae.machine
    state_enc_2 = reduce(hcat,[state_enc[:,(k-1)*tdim+2:k*tdim] for k in 1:bsize]) |> nfae.machine

    if nfae.tscale != nothing
        par_enc = reduce(hcat,[[par_enc[:,i]; nfae.tscale] for i in 1:bsize])
    end
    dz2_ = nfae.model_z.rhs(state_enc_2, par_enc*id_)
    # dx2_ = dt_NN(nfae.state.decoder,state_enc,dz2_,nfae.state.act)
    dx2_ = cat([diff(state_dec[:,:,:,(k-1)*tdim+1:k*tdim],dims=4) for k in 1:bsize]...,dims=4) |> nfae.machine 


    loss_AE_state = nfae.p_ae_state*Flux.mse(in_,state_dec)
    loss_dxdt = nfae.p_cons_x*Flux.mse(dx_, dx2_)
    loss_dzdt = nfae.p_cons_z*Flux.mse(dz_, dz2_)
    loss_AE_par = nfae.p_ae_par*Flux.mse(par_dec,par_)
    loss_AE_trans = 0.0f0
    loss_zero = 0.0f0
    if nfae.trans !=  nothing
        loss_zero = nfae.p_zero*1/nfae.model_x.x_dim*sum(
            abs,nfae.model_z.rhs(nfae.trans.encoder(par_enc[1:p_,:]),par_enc))
        dec_trans = nfae.trans.decoder(nfae.trans.encoder(par_enc[1:p_,:]))
        loss_AE_trans = nfae.p_ae_trans*Flux.mse(par_enc,dec_trans)
    else
        loss_zero = nfae.p_zero*sum(abs2,1/nfae.model_x.x_dim .* sum(state_enc, dims=2))
        loss_AE_trans = 0.0f0
    end
    loss_orient = nfae.p_orient*Flux.mae(sign.(par_enc[1:p_,:]), sign.(par_))
    nfae.loss = [loss_AE_state, loss_dxdt, loss_dzdt, loss_AE_par, loss_AE_trans, loss_zero, loss_orient]
    return loss_AE_state + loss_dxdt + loss_dzdt + loss_AE_par + loss_AE_trans + loss_zero + loss_orient
end
