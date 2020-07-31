using Pkg
Pkg.activate(".")
using DifferentialEquations
function Lorenz96!(du,u,p,t)
    du[1] = u[end]*(u[2] - u[end-1]) - u[1] + p[1]
    du[2] = u[1]*(u[3]-u[end])-u[2]+p[1]
    du[end] = u[end-1]*(u[1]-u[end-2])-u[end] + p[1]
    for i=3:(size(du,1)-1)
        du[i] = u[i-1]*(u[i+1]-u[i-2])-u[i]+p[1]
    end
end


n = 64
p = [1.0f0]
u = Float32.(rand(n))
t = (0.0f0,10.0f0)

prob = ODEProblem(Lorenz96!,u,t,p)
sol = solve(prob,Tsit5(),reltol=1e-5,abstol=1e-5)


