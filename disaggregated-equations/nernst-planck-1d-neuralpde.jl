################################################################################
#   In this code a 1D Nersnt-Plack system is solved using NeuralPDE
#
#   Equations:
#        dH/dt =  D_H * d2H/dx2 
#                 + c_H * ( dH/dx * dPhi/dx + H * d2Phi/dx2 )
#                 + k_wb * H2O - k_wf * H * OH
#        dOH/dt = D_OH * d2OH/dx2
#                 + c_OH * ( dOH/dx * dPhi/dx + OH * d2Phi/dx2 )
#                 + k_wb * H2O - k_wf * H * OH
#        Phi = Phi_0 / x_max * x
#
#   Initial conditions:
#        H(0,x) = H_0
#        OH(0,x) = OH_0
#
#   Boundary conditions:
#
#        H(t,0) = H_anode_rate * t
#        dH(t,n)/dx = 0
#        dOH(t,0)/dx = 0
#        OH(t,n) = OH_cathode_rate * t
#
#   How to run:
#        julia nernst-planck-1d-neuralpde.jl
#
################################################################################

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux

@parameters t, x
@variables H(..), OH(..)
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dxx''~x

# Electrochemical parameters
t_max = 0.01 #50.0 # s
x_max = 0.01 # m
H_anode_rate = 4.2 # mol m^-3 s^-1
OH_cathode_rate = 5.0 # mol m^-3 s^-1
H_0 = 1e-4 # mol m^-3
OH_0 = 1e-4 # mol m^-3
Phi_0 = 4.0 # V
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

eqs = [
        ( Dt(H(t,x)) ~ Dxx(H(t,x)) 
                    + c_H * Dx(H(t,x)) * Phi_0 * x / x_max
                    + k_wb * H2O - k_wf * H(t,x) * OH(t,x) ),
        ( Dt(OH(t,x)) ~ Dxx(OH(t,x)) 
                    + c_H * Dx(OH(t,x)) * Phi_0 * x / x_max
                    + k_wb * H2O - k_wf * H(t,x) * OH(t,x) )
      ]

bcs = [ 
        H(0.0,x) ~ H_0,
        OH(0.0,x) ~ OH_0,
        H(t,0.0) ~ H_anode_rate * t,
        Dx(OH(t,0.0)) ~ 0.0,
        Dx(H(t,x_max)) ~ 0.0,
        OH(t,x_max) ~ OH_cathode_rate * t
      ]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,t_max),
           x ∈ IntervalDomain(0.0,x_max)]

# Discretization
dt = 2e-3
dx = 2e-4

# Neural network
dim = length(domains)
output = length(eqs)
chain = FastChain(FastDense(dim,12,Flux.σ),FastDense(12,12,Flux.σ),FastDense(12,output))

discretization = PhysicsInformedNN(chain, strategy = GridTraining(dx=[dt,dx]))

pde_system = PDESystem(eqs,bcs,domains,[t,x],[H,OH])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

#res = GalacticOptim.solve(prob,Optim.BFGS(); cb = cb, maxiters=5000)
res = GalacticOptim.solve(prob, ADAM(0.1), cb = cb, maxiters=2000)
phi = discretization.phi 

ts,xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
ts_ = ts[1:10:size(ts)[1]]
H_predict  = [ [phi([t,x],res.minimizer)[2] for x in xs] for t in ts_] 
OH_predict = [ [phi([t,x],res.minimizer)[1] for x in xs] for t in ts_] 

using Plots
p1 = plot(xs, H_predict, title = "H+");
p2 = plot(xs, OH_predict, title = "OH-");
plot(p1,p2)
savefig("nerst-planck-1d-neuralpde.svg")


