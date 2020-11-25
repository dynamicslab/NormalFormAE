function makebatch(data::Dict,batchsize, batchnum)
    bsize = batchsize
    x = nothing
    dx = nothing
    alpha = nothing
    ind = ((batchnum-1)*bsize + 1):batchnum*bsize
    x = reduce(hcat,[data["x"][:,:,i] for i in ind])
    dx = reduce(hcat,[data["dx"][:,:,i] for i in ind])
    alpha = reduce(hcat,[data["alpha"][:,i] for i in ind])
    return x, dx, alpha
end

function makebatch(file::String, shuffle_ind, batchsize, batchnum)
    ind = shuffle_ind[((batchnum-1)*bsize + 1):batchnum*batchsize]
    x = reduce(hcat,[FileIO.load(file,"x_$(i)") for i in ind])
    alpha = FileIO.load(file,"alpha")
    return x, alpha[:,ind]
end
