function rhs(x,t,p)
    return [x[2] + x[1]^2, 0]
end

function rhs(t,x::Array{T,2}) where {T<:Number}
    return [x[2,:]' .+ x[1,:]'.^2; zeros(size(x)[2])']
end

function rhs(x::Array{T,2}) where {T<:Number}
    return [x[2,:]' .+ x[1,:]'.^2; zeros(size(x)[2])']
end

# function rhs(x)
#     return [x[2] + x[1]^2, 0]
# end    
