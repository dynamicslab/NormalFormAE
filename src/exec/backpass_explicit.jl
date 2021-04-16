
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
    end

    return activation_
end

function act_der(act::String,x;avgder=0)

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
    else
        der_ = map(y->y./y,x)
    end
    return der_
end

function act_fun(act,x)
    if act == "id"
        return x
    else
        return get_activation(act).(x)
    end
end

function dt_NN(NN, input_, left_dt, act)
    # Given input, left_dt and neural net NN, compute right hand side time-derivative via chain rule
    # As per Champion et al. (PNAS 2019)
    # Neural Net is of the form Chain(Dense(_,_,act),...,Dense(_,_))
      
    # Handle dropout layers
    start_ind = 1
    if isempty(Flux.params(NN.layers[1])) # it's a dropout
       start_ind = 2
    end   
    W_,b_ = Flux.params(NN.layers[start_ind])
    l = W_*input_.+b_
    dl = W_*left_dt
    nlayers = length(NN.layers) # 2x number of layers, to handle dropouts
    for i=(start_ind+1):nlayers
        if !isempty(Flux.params(NN.layers[i]))
            W_,b_ = Flux.params(NN.layers[i])
            if act != "id"
        	dl = W_*(act_der(act,l).*dl)
            else
                dl = W_*dl
            end
            l = W_*act_fun(act,l) .+ b_
        end
    end     
    return dl
end
