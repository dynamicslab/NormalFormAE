function construct(in_dim, out_dim, widths_, act_)
    w = deepcopy(widths_)
    NN = []
    pushfirst!(w, in_dim)
    push!(w, out_dim)
    if act_ != nothing
        act_fun = Flux.eval(act_)
    else
        act_fun = []
    end
    i = 1
    while i < length(w)
        i == (length(w)-1) ? activation = [] : activation = act_fun
        args = [w[i]; w[i+1]; activation]
        push!(NN,Dense(args...))
        i = i +1
    end
    return NN
end
function gen_fluid_operator(nfae::NFAE{xname,zname},lift,drop,state_widths, par_widths) where {xname,zname}
    if nfae.state.act == "id"
        act_state = nothing
    else
        act_state = Symbol(nfae.state.act)
    end
    if nfae.par.act == "id"
        act_par = nothing
    else
        act_par = Symbol(nfae.par.act)
    end

    nfae.state = AE(:State, nfae.state.out_dim,nfae.state.out_dim,state_widths,act_state,nfae.machine) # respawn
    nfae.par = AE(:Par, nfae.par.in_dim,nfae.par.out_dim,par_widths,act_par,nfae.machine) # respawn

    nfae.state.encoder = Chain
    

