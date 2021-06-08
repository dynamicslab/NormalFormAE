using Pkg
# Make sure pwd() evaluates to NormalFormAE
Pkg.activate("$(pwd())/fluid_data/.")
using ViscousFlow, LinearAlgebra, JLD2
using FileIO

Re = 80; # Reynolds number

Δx, Δt = setstepsizes(Re,gridRe=2)


tfinal = 77.0
# tsize = Int(div(tfinal,Δt)) + 2
x_size = 486
y_size = 250
vec_size = x_size*y_size
svd_index = 325
nmodes = 10

Re_range = Array(range(30,70.0f0,length=239)); # Reynolds number


FileIO.save("$(pwd())/fluid_data/pod_data.jld2","Re",Array(Re_range))

for (idx,Re) in enumerate(Re_range)
    if idx>0
    ## STEP 1: Simulate Navier Stokes over high fidelity grid
    U = 1.0; # Free stream velocity
    U∞ = (U,0.0);
    xlim = (-2.0,10.0)
    ylim = (-3.0,3.0);
    body = Circle(0.5,Δx)
    sys = NavierStokes(Re,Δx,xlim,ylim,Δt,body,freestream = U∞)
    u0 = newstate(sys);
    for i in 1:vec_size
        u0[i] = u0[i] + 0.01*rand()
    end
    tspan = (0.0,tfinal)
    tt = range(0.0,tfinal,length=1000)
    integrator = init(u0,tspan,sys)
    @time step!(integrator,tfinal)
    
    vec_ = reduce(hcat, [Array(integrator.sol.u[i]) for i in 1:10:size(integrator.sol.t)[1]])
    vec_dim = size(vec_)

    ## STEP 3: SVD

    tmp = vec_[:,svd_index];
    for i = (svd_index+1):vec_dim[2]   
        tmp = tmp .+ vec_[:,i]
    end
    vec_avg = tmp./(vec_dim[2]-svd_index + 1)
    vec_svd = vec_[:,svd_index:end] .- vec_avg
    s = svd(vec_svd)
    JLD2.jldopen("$(pwd())/fluid_data/pod_data.jld2", "a+") do file
       file["x_$(180 + idx)_xmode"] = s.U[:,1:nmodes]
       file["x_$(180 + idx)_sing"] = s.S[1:nmodes]
       file["x_$(180 + idx)_tmode"] = s.Vt[1:nmodes,:]
    end
    s = nothing
    vec_ = nothing
    sol = nothing
    integrator = nothing

    println("Done: $(idx)")
end
end


#Re_range_orig = Array(range(30.0f0,70.0f0,length=120))
#Re_range = filter(x->!(x in Re_range_orig),Re_range); 
#
#function get_alpha(ind_)
#   if ind_< 121
#       return range(30.0f0,70.0f0,length=120)[ind_]
#   else
#       aa = range(30.0f0,70.0f0,length=239) |> Array
#       bb = range(30.0f0,70.0f0,length=120) |> Array
#       cc = filter(x->!(x in bb),aa)
#       return cc[ind_-120]
#   end
#end
#
#Re_range = [get_alpha(3), get_alpha(115)]
