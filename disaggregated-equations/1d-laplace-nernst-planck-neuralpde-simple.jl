#################################################################################
#   This code solves a simplified 1D Laplace-Nernst-Planck system using NeuralPDE
#
#   Equations:
#        dH/dt =  D_H * d2H/dx2 
#                 + c_H * ( dH/dx * dPhi/dx + H * d2Phi/dx2 )
#        dOH/dt = D_OH * d2OH/dx2
#                 + c_OH * ( dOH/dx * dPhi/dx + OH * d2Phi/dx2 )
#        Phi = Phi_0 / x_max * x
#
#       Notes:  Laplace equation is solved analytically
#               Reaction term is neglected
#
#   Initial conditions:
#        H(0,x) = H_0
#        OH(0,x) = OH_0
#
#   Boundary conditions:
#        H(t,0) = H_anode_rate
#        H(t,n) = H_0
#        OH(t,0) = OH_0
#        OH(t,n) = OH_cathode_rate
#
#   How to run:
#        julia 1d-laplace-nernst-planck-neuralpde-simple.jl
#
#################################################################################

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature,Cubature, Cuba

@parameters t,x
@variables H(..),OH(..)
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dxx''~x

# Electrochemical parameters
t_max = 0.01 #50.0 # s
x_max = 0.1 # m
H_anode_rate = 4.2 # mol m^-3 s^-1
OH_cathode_rate = 5.0 # mol m^-3 s^-1
H_0 = 1e-4 # mol m^-3
OH_0 = 1e-4 # mol m^-3
Phi_0 = 2.0 # V
z_H = 1.0
D_H = 9.31e-9 # m^2 s^-1
z_OH = -1.0
D_OH = 5.26e-9 # m^2 s^−1 
F = 96485.3415 # A s mol^-1
R = 8.31 # kg m^2 K^-1 mol^-1 s^-2
T = 298.0 # K
σ_0 = 0.2 # S m^-1
k_wf = 1.5e8 # m^3 mol^-1 s^-1
k_wb = 2.7e-5 # s^-1
H2O = 55500.0 # mol m^-3
c_H = z_H * D_H * F / (R * T) 
c_OH = z_OH * D_OH * F / (R * T) 

# Equations, initial and boundary conditions

#eqs = [
#        ( Dt(H(t,x)) ~ D_H * Dxx(H(t,x)) 
#                     + c_H * Dx(H(t,x)) * Phi_0 * (x_max-x) / x_max
#                     + k_wb * H2O - k_wf * H(t,x) * OH(t,x) ),
#        ( Dt(OH(t,x)) ~ D_OH * Dxx(OH(t,x)) 
#                     + c_H * Dx(OH(t,x)) * Phi_0 * (x_max-x) / x_max
#                     + k_wb * H2O - k_wf * H(t,x) * OH(t,x) )
#      ]

#bcs = [ 
#        H(0.0,x) ~ H_0,
#        OH(0.0,x) ~ OH_0,
#        H(t,0.0) ~ H_anode_rate * t + H_0,
#        OH(t,0.0) ~ OH_0,
#        H(t,x_max) ~ H_0,
#        OH(t,x_max) ~ OH_cathode_rate * t + OH_0
#      ]


# Simplified system
eqs = [
        ( Dt(H(t,x)) ~ D_H * Dxx(H(t,x)) 
                     + c_H * Dx(H(t,x)) * Phi_0 * (x_max-x) / x_max),
        ( Dt(OH(t,x)) ~ D_OH * Dxx(OH(t,x)) 
                     + c_H * Dx(OH(t,x)) * Phi_0 * (x_max-x) / x_max)
      ]

bcs = [ 
        H(0.0,x) ~ H_0,
        OH(0.0,x) ~ OH_0,
        H(t,0.0) ~ H_anode_rate,
        OH(t,0.0) ~ OH_0,
        H(t,x_max) ~ H_0,
        OH(t,x_max) ~ OH_cathode_rate
      ]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,t_max),
           x ∈ IntervalDomain(0.0,x_max)]

# Discretization
dt = 5e-9
dx = 1e-3

# Neural network
dim = length(domains)
output = length(eqs)
neurons = 12
chain1 = FastChain( FastDense(dim,neurons,Flux.σ),
                    FastDense(neurons,neurons,Flux.σ),
                    FastDense(neurons,1))
chain2 = FastChain( FastDense(dim,neurons,Flux.σ),
                    FastDense(neurons,neurons,Flux.σ),
                    FastDense(neurons,1))

#strategy = GridTraining(dx=[dt,dx])
strategy = QuadratureTraining(algorithm=HCubatureJL(),reltol= 1e-5,abstol= 1e-5,maxiters=500)
#strategy = StochasticTraining(number_of_points = 100)

discretization = PhysicsInformedNN([chain1,chain2],strategy=strategy)

pde_system = PDESystem(eqs,bcs,domains,[t,x],[H,OH])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob,Optim.BFGS();cb=cb,maxiters=1000)
#res = GalacticOptim.solve(prob, ADAM(0.001), cb = cb, maxiters=1000)

phi = discretization.phi

initθ = discretization.initθ
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers = [res.minimizer[s] for s in sep]

ts,xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
ts_ = ts[1:10:size(ts)[1]]
H_predict = [ [ phi[1]([t,x],minimizers[1])[1] for x in xs] for t in ts_] 
OH_predict = [ [ phi[2]([t,x],minimizers[2])[1] for x in xs] for t in ts_] 

using Plots, LaTeXStrings
p1 = plot(xs, H_predict, title = L"$H^+ | mol \  m^{-3}$");
p2 = plot(xs, OH_predict, title = L"$OH^- | mol \ m^{-3}$");
plot(p1,p2)
savefig("1d-laplace-nerst-planck-neuralpde-simple.svg")


