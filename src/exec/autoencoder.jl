
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

function act_der2(act::String,x,avgder=0)
    if act == "sigmoid"
        der_1 = sigmoid.(x) .*(1.0f0 .- sigmoid.(x))
        der_ = der_1 .*( 1.0f0 .- 2.0f0 .* sigmoid.(x) )
    elseif act == "tanh"
        der_1 = 1.0f0 .- (tanh.(x)).^2
        der_ = 1.0f0 .- 2.0f0 .* tanh.(x) .* der_1
    elseif act == "relu"
        der_ = 0.0f0 .* x
    elseif act == "leakyrelu"
        der_ = 0.0f0 .* x
    elseif act == "elu"
        der_ = -exp.(x)/2.0f0 .* (sign.(x) .- abs.(sign.(x)))
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
    tscale = [args["tscale_init"]]
 
    return encoder, decoder, par_encoder,par_decoder, tscale
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

function d2t_NN(NN,input_, acts, i, j)
    W_,b_ = Flux.params(NN.layers[1])
    l = W_*input_.+b_
    dxi = W_*gpu(Matrix{Float32}(I,size(input_)[1],size(input_)[1]))[:,i]
    dxj = W_*gpu(Matrix{Float32}(I,size(input_)[1],size(input_)[1]))[:,j]
    dxidxj = 0.0f0
    act = acts[1]
    nlayers = div(length(Flux.params(NN)),2)
    for k = 2:nlayers
        W_,b_ = Flux.params(NN.layers[k])
        act_fun = get_activation(act)
        if act != "id"
            actder = act_der(act,l)
            actder2 = act_der2(act,l)
            dxidxj = W_*(actder2 .* dxi .*dxj .+ actder .* dxidxj )
            dxi = W_*(actder .* dxi)
            dxj = W_*(actder .* dxj)
            l = W_*act_fun.(l) .+ b_
            act = acts[k]
        else
            dxidxj = W_ * (dxidxj)
            dxi = W_*(dxi)
            dxj = W_*dxj
            l = W_*x + b_
            act = acts[k]
        end
    end
    if act != "id"        
        act_fun = get_activation(act)
        actder = act_der(act,l)
        actder2 = act_der2(act,l)
        dxidxj = actder2 .* dxi .*dxj .+ actder .* dxidxj
        dxi = actder .* dxi
        dxj = actder .* dxj
    end
    return dxidxj
end


function hess_vec(args,NN,input_,acts, xb,xt)
    # Assumption: xb \in R^(n x t x p), xt \in R^(n x t)
    tdim = args["tsize"]
    n_ = args["x_dim"]
    p_ = args["par_dim"]
    dxx_temp = reshape(xb, args["x_dim"],tdim*p_)
    dxx = 0.0f0 .* NN(dxx_temp)
    for j = 1:n_
        for k = 1:n_
            id_4 = hcat([Matrix{Float32}(I,args["tsize"],args["tsize"]) for i in 1:p_]...) |> gpu
            dxidxj = d2t_NN(NN,input_,acts,j,k)*id_4 # (n,t x p)
            prod_ = reshape(xb[k,:,:] .* xt[j,:],1,tdim*p_) # (1, t x p)
            dxx = dxx .+ dxidxj.*prod_ # (n, t x p)
        end
    end
    return dxx
end

function build_loss(args,dzdt_rhs,dzdt_solve,dzdt_sens_rhs)    
    t_batch = range(0.0f0,Float32(args["tspan"][2]),length = args["tsize"])
    ode_prob_temp = ODEProblem(dzdt_solve,rand(Float32,args["z_dim"]),(0.0f0,Float32(args["tspan"][2])),gpu(rand(args["par_dim"])))
    ode_prob = (x,y) -> remake(ode_prob_temp,p=x,u0=y)
    #solve(ode_prob(gpu(rand(1))))
    #ODE solve to be used for training
    function predict_ODE_solve(x,y)
        return Array(solve(ode_prob(x,y),Tsit5(),saveat=t_batch,reltol=1e-4)) 
    end
    function loss_(encoder, decoder, par_encoder, par_decoder, u0_train,tscale,in_batch,dx_batch,par_batch,dxda_batch,dtdxda_batch,testt)
        bsize = 0
        if testt>0
            bsize = args["test_size"]
        else
            bsize = args["BatchSize"]
        end
        tdim = args["tsize"]
        n_ = args["x_dim"]
        p_ = args["par_dim"]
        # loss_AE_state = 0.0f0
        # loss_dxdt = 0.0f0
        # loss_dzdt = 0.0f0
        # loss_AE_par = 0.0f0
        loss_sens_dtdzdb = 0.0f0
        loss_sens_x = 0.0f0
        loss_par_id = 0.0f0
        # loss_NLRAN_in = 0.0f0
        # loss_NLRAN_out = 0.0f0
        # loss_u0 = 0.0f0

        in_ = in_batch
        dx_ = dx_batch
        par_ = par_batch
        
        enc_ = encoder(in_)
        par_adjust = par_encoder(par_)
        enc_par = par_adjust # CAREFUL: TRANSLATION ADDED
        dec_par = par_decoder(par_adjust) # CAREFUL: TRANSLATION ADDED
        dec_ = decoder(enc_)
        # Consistency losses
        dz1 = dt_NN(encoder,in_,dx_,args["AE_acts"])
        #enc_par_temp = enc_par*reduce(hcat,[Matrix{Float32}(I,1,1) for i in 1:tdim])
        id_ = repeat(Matrix{Float32}(I,bsize,bsize),inner=(1,args["tsize"])) |> gpu
        enc_par_aug = enc_par
        if args["Par_widths"][end]-args["Par_widths"][1] == 0
            enc_par_aug = hcat([[enc_par[:,i]; tscale] for i in 1:bsize]...) |> gpu
            if args["P_NLRAN_in"] != 0
                enc_ODE_solve = hcat([predict_ODE_solve([enc_par[:,i]; tscale],u0_train[:,i]) for i in 1:bsize]...) |> gpu
            else
                enc_ODE_solve = 0.0f0
            end
        else
            if args["P_NLRAN_in"] != 0
                enc_ODE_solve = hcat([predict_ODE_solve(enc_par[:,i],u0_train[:,i]) for i in 1:bsize]...) |> gpu
            else
                enc_ODE_solve = 0.0f0
            end
        end
        dx1 = dt_NN(decoder,enc_,dzdt_rhs(enc_,enc_par_aug*id_,0.0f0,bsize),[reverse(args["AE_acts"])[2:end];"id"])
        # NLRAN losses
        
        # enc_ODE_solve = hcat([predict_ODE_solve(enc_par[:,i],u0_train[:,i]) for i in 1:bsize]...) |> gpu
        #u0 loss
        #enc_init = hcat([enc_[:,(i-1)*args["tsize"]+1] for i in 1:bsize]...) |> gpu

        
        # # AE loss
        # for i in 1:args["BatchSize"]

        #     in_ = in_batch[:,:,i]
        #     dx_ = dx_batch[:,:,i]
        #     par_ = par_batch[:,i]
        #     #dxda = dxda_batch[:,:,:,i]
        #     #dtdxda = dtdxda_batch[:,:,:,i]
            
        #     enc_ = encoder(in_)
        #     par_adjust = par_encoder(par_)
        #     enc_par = par_adjust # CAREFUL: TRANSLATION ADDED
        #     dec_par = par_decoder(par_adjust) # CAREFUL: TRANSLATION ADDED
        #     dec_ = decoder(enc_)
        #     # Consistency losses
        #     dz1 = dt_NN(encoder,in_,dx_,args["AE_acts"])
        #     #enc_par_temp = enc_par*reduce(hcat,[Matrix{Float32}(I,1,1) for i in 1:tdim])
        #     dx1 = dt_NN(decoder,enc_,dzdt_rhs(enc_,enc_par,0.0f0),[reverse(args["AE_acts"])[2:end];"id"])
        #     # NLRAN losses
        #     enc_ODE_solve = predict_ODE_solve(enc_par,u0_train[:,i])
            
            
            
        #     # Sensitivity losses
        #     # 1. Compute dxdb and dtdxdb
        #     # dxda_ = reshape(dxda,n_*tdim,p_)
        #     # dtdxda_ = reshape(dtdxda,n_*tdim,p_)
        #     # id_ = hcat([Matrix{Float32}(I,1,1) for i in 1:p_]...) |> gpu
        #     # p_temp = enc_par*id_
        #     # id_p = Matrix{Float32}(I,p_,p_) |> gpu
        #     # dxdb_ = dxda_*dt_NN(par_decoder,p_temp,id_p,[reverse(args["Par_acts"])[2:end];"id"])
        #     # dxdb = reshape(dxdb_,n_,tdim,p_)
        #     # dtdxdb_ = dtdxda_*dt_NN(par_decoder,p_temp,id_p,[reverse(args["Par_acts"])[2:end];"id"])
        #     # dtdxdb = reshape(dtdxdb_,n_,tdim,p_)
        #     # # 2. dzdb and dtdzdb
        #     # dxdb_ = reshape(dxdb,n_,tdim*p_)
        #     # dtdxdb_ = reshape(dtdxdb,n_,tdim*p_)
        #     # id2_ = hcat([Matrix{Float32}(I,args["tsize"],args["tsize"]) for i in 1:p_]...) |> gpu
        #     # x_temp = in_*id2_
        #     # dzdb_ = dt_NN(encoder,x_temp,dxdb_,args["AE_acts"])
        #     # dtdzdb_2 = dt_NN(encoder,x_temp,dtdxdb_,args["AE_acts"])
        #     # dtdzdb_1 = hess_vec(args,encoder,in_,args["AE_acts"],dxdb,dx_)
        #     # dtdzdb_ = dtdzdb_1 .+ dtdzdb_2
        #     # dtdzdb = reshape(dtdzdb_,args["z_dim"],tdim,p_)
        #     # dzdb = reshape(dzdb_,args["z_dim"],tdim,p_)
        #     # # 3. Compute dtdzdb_data
        #     # dtdzdb_data_ = vcat([dzdt_sens_rhs(enc_[:,i],enc_par,0.0f0,dzdb[:,i,:]) for i in 1:tdim]...) |> gpu
        #     # dtdzdb_data = reshape(dtdzdb_data_,args["z_dim"],tdim,p_)
        #     # # 4. Compute dxda_decode
        #     # dxda_ = reshape(dxda,n_,tdim*p_)
        #     # id_3 = hcat([Matrix{Float32}(I,args["tsize"],args["tsize"]) for i in 1:p_]...) |> gpu
        #     # z_temp = enc_*id_3
        #     # dxda_decode_1 = dt_NN(encoder, x_temp, dxda_, args["AE_acts"])
        #     # dxda_decode_ = dt_NN(decoder, z_temp, dxda_decode_1, [reverse(args["AE_acts"])[2:end];"id"] )
        #     # dxda_decode = reshape(dxda_decode_,n_,tdim,p_)
            
            
        #     loss_AE_state = loss_AE_state + Float32(1/args["BatchSize"])*args["P_AE_state"]*Flux.mse(in_,dec_)
        #     loss_dxdt = loss_dxdt + Float32(1/args["BatchSize"])*args["P_cons_x"]*Flux.mse(dx_,dx1)
        #     loss_dzdt = loss_dzdt + Float32(1/args["BatchSize"])*args["P_cons_z"]*Flux.mse(dz1,dzdt_rhs(enc_,enc_par,0.0f0))
        #     loss_AE_par = loss_AE_par + Float32(1/args["BatchSize"])*args["P_AE_par"]*Flux.mse(dec_par,par_)
        #     loss_NLRAN_in = loss_NLRAN_in + Float32(1/args["BatchSize"])*args["P_NLRAN_in"]*Flux.mse(enc_ODE_solve,enc_)
        #     loss_NLRAN_out = loss_NLRAN_out + Float32(1/args["BatchSize"])*args["P_NLRAN_out"]*Flux.mse(in_,decoder(enc_ODE_solve))
        #     loss_u0 = loss_u0 + Float32(1/args["BatchSize"])*args["P_u0"]*Flux.mse(u0_train[:,i],enc_[:,1])
        #     # loss_sens_dtdzdb = loss_sens_dtdzdb + Float32(1/args["BatchSize"])*args["P_sens_dtdzdb"]*Flux.mse(dtdzdb,dtdzdb_data)
        #     # loss_sens_x = loss_sens_x + Float32(1/args["BatchSize"])*args["P_sens_x"]*Flux.mse(dxda,dxda_decode)
        #     # loss_par_id = loss_par_id + Float32(1/args["BatchSize"])*args["P_AE_id"]*Flux.mse(par_,par_adjust)
        # end
        loss_AE_state = args["P_AE_state"]*Flux.mse(in_,dec_)
        loss_dxdt = args["P_cons_x"]*Flux.mae(dx_,dx1)
        loss_dzdt = args["P_cons_z"]*Flux.mae(dz1,dzdt_rhs(enc_,enc_par_aug*id_,0.0f0,bsize))
        loss_AE_par = args["P_AE_par"]*Flux.mse(dec_par,par_)
        if args["P_NLRAN_in"] !=0
            loss_NLRAN_in = args["P_NLRAN_in"]*Flux.mse(enc_ODE_solve,enc_)
            loss_NLRAN_out = args["P_NLRAN_out"]*Flux.mse(in_,decoder(enc_ODE_solve))
        else
            loss_NLRAN_in = 0.0f0
            loss_NLRAN_out = 0.0f0
        end
        #loss_u0 = args["P_u0"]*Flux.mse(u0_train,enc_init)
        loss_orient = args["P_orient"]*Flux.mae(sign.(enc_par_aug[1:p_,:]) , sign.(par_))
        loss_zero = args["P_zero"]*1/args["x_dim"]*sum(abs,encoder(gpu(zeros(Float32,args["x_dim"],1))))

        
        args["rel_loss_AE"] = loss_AE_state/sum(abs2,in_)*bsize
        args["rel_loss_dxdt"] = loss_dxdt/sum(abs2,dx_)*bsize
        args["rel_loss_dzdt"] = loss_dzdt/sum(abs2,dz1)*bsize
        args["rel_loss_par"] = loss_AE_par/sum(abs2,par_)*bsize
        args["rel_loss_sens_dt"] = loss_sens_dtdzdb
        args["rel_loss_sens_x"] = loss_sens_x
        args["rel_loss_par_id"] = loss_par_id
        if args["P_NLRAN_in"] !=0
            args["rel_loss_NLRAN_in"] = loss_NLRAN_in/sum(abs2,enc_ODE_solve)*bsize
            args["rel_loss_NLRAN_out"] = loss_NLRAN_out/sum(abs2,in_)*bsize
        else
            args["rel_loss_NLRAN_in"] = 0.0f0
            args["rel_loss_NLRAN_out"] = 0.0f0
        end
        args["rel_loss_orient"] = loss_orient/sum(abs2,sign.(par_))*bsize
        args["rel_loss_zero"] = loss_zero

        args["loss_AE"] = loss_AE_state
        args["loss_dxdt"] = loss_dxdt
        args["loss_dzdt"] = loss_dzdt
        args["loss_par"] = loss_AE_par
        args["loss_sens_dt"] = loss_sens_dtdzdb
        args["loss_sens_x"] = loss_sens_x
        args["loss_par_id"] = loss_par_id
        args["loss_NLRAN_in"] = loss_NLRAN_in
        args["loss_NLRAN_out"] = loss_NLRAN_out
        args["loss_orient"] = loss_orient
        args["loss_zero"] = loss_zero
        
        
        loss_total = loss_AE_state + loss_dxdt + loss_dzdt + loss_AE_par + loss_NLRAN_in + loss_NLRAN_out + loss_orient + loss_zero
        args["loss_total"] = loss_total
        args["rel_loss_total"] = args["rel_loss_AE"] +args["rel_loss_dxdt"] + args["rel_loss_dzdt"]+ args["rel_loss_par"] + args["rel_loss_sens_dt"] + args["rel_loss_sens_x"] + args["rel_loss_par_id"] +  args["rel_loss_NLRAN_in"] + args["rel_loss_NLRAN_out"] + args["rel_loss_orient"]+ args["rel_loss_zero"] 
        return loss_total
    end
    return loss_
end

        
        
        
    
        
