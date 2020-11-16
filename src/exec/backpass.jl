# Needs Zygote, LinearAlgebra
function dt_NN_single(NN, in_, din_dt)
    # As per Champion et al. (PNAS, 2019)
    # Uses pullback of pullback
    res, back = Zygote.pullback(NN,in_)
    len = length(res)
    mat = Matrix{Float32}(I,len,len) |> gpu
    jac = reduce(hcat,[back(gpu(mat[:,i]))[1] for i in 1:len])
    #dout_dt = transpose(jac)*din_dt
    dout_dt = jac
end

function dt_NN(NN, in_, din_dt)
    dout_dt = reduce(hcat,[dt_NN_single(NN, in_[:,i], din_dt[:,i]) for i in 1:size(in_)[end]])
end
