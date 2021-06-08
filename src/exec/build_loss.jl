function (nfae::NFAE{xname,zname})(in_, dx_, par_) where {xname,zname}
    
    bsize = div(size(in_)[end], nfae.model_x.tsize)
    tdim = nfae.model_x.tsize
    n_ = nfae.model_x.x_dim
    p_ = nfae.model_z.par_dim
    
    state_enc = nfae.state.encoder(in_)

    par_enc = nfae.par.encoder(par_)
    id_ = repeat(Matrix{Float32}(I,bsize,bsize),inner=(1,nfae.model_x.tsize)) |> nfae.machine
    par_dec = nfae.par.decoder(par_enc)
    state_dec = nfae.state.decoder(state_enc)

    # Consistency losses
    dz_ = dt_NN(nfae.state.encoder,in_,dx_,nfae.state.act)
    par_enc2 = reduce(hcat,[[par_enc[:,i]; nfae.tscale] for i in 1:bsize])
    dz2_ = nfae.model_z.rhs(state_enc, par_enc2*id_)
    dx2_ = dt_NN(nfae.state.decoder,state_enc,dz2_,nfae.state.act)

    loss_AE_state = nfae.p_ae_state*Flux.mse(in_./in_,state_dec./in_) + Flux.mse(in_,state_dec)
    loss_dxdt = nfae.p_cons_x*Flux.mse(dx_, dx2_)
    loss_dzdt = nfae.p_cons_z*Flux.mse(dz_, dz2_)
    loss_AE_par = nfae.p_ae_par*Flux.mse(par_dec,par_)
    loss_zero = nfae.p_zero*(1/bsize)*sum(abs2,(1/nfae.model_x.x_dim) .* sum(state_enc, dims=2))
    # loss_zero = nfae.p_zero*Flux.mse(nfae.state.encoder(0.0f0 .* in_),0.0f0 .* in_)
    loss_orient = nfae.p_orient*Flux.mse(sign.(par_enc[1:p_,:]), sign.(par_))
    nfae.loss = [loss_AE_state, loss_dxdt, loss_dzdt, loss_AE_par,  loss_orient, loss_zero]
    return loss_AE_state + loss_dxdt + loss_dzdt + loss_AE_par  + loss_orient + loss_zero
end
