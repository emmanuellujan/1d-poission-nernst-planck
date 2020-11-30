################################################################################
#   In this code a 1D Nersnt-Plack/Laplace system is solved using NeuralPDE
#
#   Equations:
#        d2φ/dx2 = 0
#        dH/dt =  D_H * d2H/dx2 
#                 + c_H * ( dH/dx * dφ/dx + H * d2φ/dx2 )
#                 + k_wb * H2O - k_wf * H * OH
#        dOH/dt = D_OH * d2OH/dx2
#                 + c_OH * ( dOH/dx * dφ/dx + OH * d2φ/dx2 )
#                 + k_wb * H2O - k_wf * H * OH
#
#   Initial condition: 
#        H(0,x) = H_0
#        OH(0,x) = OH_0
#        φ(0,x) = φ_0
#
#   Boundary conditions:
#        H(t,0) = H_anode_rate * t
#        H(t,n) = H_0
#        OH(t,0) = OH_0
#        OH(t,n) = OH_cathode_rate * t
#        φ(t,0) = φ_0
#        φ(t,n) = 0
#
#   How to run:
#        julia nernst-planck-neural-pde.jl
#
################################################################################

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux

@parameters t, x
@variables H(..), OH(..), φ(..)
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dxx''~x

# Electrochemical parameters
t_max = 50
x_max = 0.01
H_anode_rate = 4.2 # mol m^-3 s^-1
OH_cathode_rate = 5 # mol m^-3 s^-1
H_0 = 1e-4 # mol m^-3
OH_0 = 1e-4 # mol m^-3
φ_0 = 4 # V
z_H = 1
D_H = 9.31e-9 # m^2 s^-1
z_OH = -1
D_OH = 5.26e-9 # m^2 s^−1 
F = 96485.3415 # A s mol^-1
R = 8.31 # kg m^2 K^-1 mol^-1 s^-2
T = 298 # K
σ_0 = 0.2 # S m^-1
k_wf = 1.5e8 # m^3 mol^-1 s^-1
k_wb = 2.7e-5 # s^-1
H2O = 55500 # mol m^-3
c_H = z_H * D_H * F / (R * T) 
c_OH = z_OH * D_OH * F / (R * T) 

# Equations, initial and boundary conditions
#eqs = [
#        0. ~ σ_0 * Dxx(φ(t,x)),
#        Dt(H(t,x)) ~ Dxx(H(t,x)) 
#                    + c_H * Dx(H(t,x)) * φ(t,x) + c_H * H(t,x) * Dxx(φ(t,x))
#                    + k_wb * H2O - k_wf * H(t,x) * OH(t,x),
#        Dt(OH(t,x)) ~ Dxx(OH(t,x)) 
#                    + c_H * Dx(OH(t,x)) * φ(t,x) + c_H * OH(t,x) * Dxx(φ(t,x))
#                    + k_wb * H2O - k_wf * H(t,x) * OH(t,x)
#      ]
eqs = [
        0. ~ σ_0 * Dxx(φ(t,x)),
        Dt(H(t,x)) ~ Dxx(H(t,x)) + c_H * Dx(H(t,x)) * φ(t,x) + c_H * H(t,x) * Dxx(φ(t,x)) + k_wb * H2O - k_wf * H(t,x) * OH(t,x),
        Dt(OH(t,x)) ~ Dxx(OH(t,x)) + c_H * Dx(OH(t,x)) * φ(t,x) + c_H * OH(t,x) * Dxx(φ(t,x)) + k_wb * H2O - k_wf * H(t,x) * OH(t,x)
      ]

bcs = [ H(0,x) ~ H_0,
        OH(0,x) ~ OH_0,
        φ(0,x) ~ φ_0,
        H(t,0) ~ H_anode_rate * t,
        OH(t,0) ~ H_anode_rate * t,
        φ(t,0) ~ φ_0,
        H(t,0.01) ~ H_0,
        OH(t,0.01) ~ OH_0,
        φ(t,0.01) ~ 0]


# Space and time domains
domains = [t ∈ IntervalDomain(0.0,t_max),
           x ∈ IntervalDomain(0.0,x_max)]

# Discretization
dx = 2e-4

# Neural network
dim = length(domains)
output = length(eqs)
chain = FastChain(FastDense(dim,16,Flux.σ),FastDense(16,16,Flux.σ),FastDense(16,output))

strategy = GridTraining()
discretization = PhysicsInformedNN(dx,chain,strategy=strategy)

pde_system = PDESystem(eqs,bcs,domains,[x,t],[H,OH,φ])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, Optim.BFGS(); cb = cb, maxiters=5000)

phi = discretization.phi





