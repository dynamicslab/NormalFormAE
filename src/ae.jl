mutable struct AE{name_}
    widths :: Array{Int64,1}
    act :: String
    in_dim :: Int64
    out_dim :: Int64
    encoder :: Any
    decoder :: Any 
    machine :: Any # gpu/cpu
end

function AE(name,in_,out_,widths, act::Union{Symbol,Nothing}, machine)
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
        Chain(NN...)
    end
    encoder = construct(in_, out_, widths, act) |> machine
    decoder = construct(out_, in_, reverse(widths), act) |> machine
    if act != nothing
        AE{name}(widths,String(act),in_, out_,encoder, decoder,machine)
    else
        AE{name}(widths,"id",in_, out_,encoder, decoder,machine)
    end
end

Base.show(io::IO, NN::AE{name_}) where name_  =
    print(io, String(name_),
          " Autoencoder with (forward) hidden widths ",NN.widths, " and activation ",NN.act)
