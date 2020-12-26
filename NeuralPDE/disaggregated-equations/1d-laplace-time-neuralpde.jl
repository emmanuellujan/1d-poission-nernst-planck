################################################################################
#   In this code a 1D laplace eq. is solved using NeuralPDE
#
#   Equations:
#        d2Phi/dx2 = 0
#
#   Boundary conditions:
#
#        Phi(t,0) = Phi_0
#        Phi(t,n) = 0
#
#   How to run:
#        julia laplace-1d-time-neuralpde.jl
#
################################################################################

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux

@parameters t, x
@variables Phi(..)
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
        Dxx(Phi(t,x)) ~ 0.0
      ]

bcs = [ 
        Phi(t,0.0) ~ Phi_0,
        Phi(t,x_max) ~ 0.0
      ]

# Space and time domains
domains = [
            t ∈ IntervalDomain(0.0,t_max),
            x ∈ IntervalDomain(0.0,x_max)
          ]

# Discretization
dt = 2e-3
dx = 2e-4

# Neural network
dim = length(domains)
output = length(eqs)
chain = FastChain(FastDense(dim,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,output))

discretization = PhysicsInformedNN(chain, strategy = GridTraining(dx=[dt,dx]))

pde_system = PDESystem(eqs,bcs,domains,[t, x],[Phi])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, Optim.BFGS(); cb = cb, maxiters=200)

phi = discretization.phi

ts,xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains]

u_predict = reshape([first(phi([x,t],res.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))

using Plots
plot(xs, ts, u_predict, st=:contourf,title = "predict");
savefig("laplace-1d-time-neuralpde.svg")

