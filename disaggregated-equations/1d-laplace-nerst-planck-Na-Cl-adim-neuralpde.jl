#################################################################################
#   This code solves a simplified 1D Laplace-Nernst-Planck system using NeuralPDE
#
#   Equations:
#        dH/dt =  (1.0/Pe_Na) * d2Na/dx2 
#                 + z_Na / (abs(z_Na)*M_Na) * ( dNa/dx * dPhi/dx + Na * d2Phi/dx2 )
#        dCl/dt = (1.0/Pe_Cl) * d2Cl/dx2
#                 + z_Cl / (abs(z_Na)*M_Na) * ( dCl/dx * dPhi/dx + Cl * d2Phi/dx2 )
#        Phi = Phi_0 / x_max * x
#
#       Notes:  Laplace equation is solved analytically 
#
#   Initial conditions:
#        Na(0,x) = Na_0
#        Cl(0,x) = Cl_0
#
#   Boundary conditions:
#        Na(t,0) = Na_anode_rate
#        Na(t,n) = Na_cathode_rate
#        Cl(t,0) = Cl_anode_rate
#        Cl(t,n) = Cl_cathode_rate
#
#   How to run:
#        julia 1d-laplace-nernst-planck-neuralpde-Na-Cl-adim.jl
#
#################################################################################

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature,Cubature, Cuba

@parameters t,x
@variables Na(..),Cl(..)
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dxx''~x

# Electrochemical parameters

z_H = 1.0
z_OH = -1.0
z_Na = 1.0
z_Cl = -1.0

D_H = 9.31e-5 # cm^2 s^-1
D_OH = 5.26e-5 # cm^2 s^−1 
D_Na = 1.33e-5 # cm^2 s^−1
D_Cl = 2.03e-5 # cm^2 s^−1

u_H = 3.24e-4 # cm^2 V^-1 s^-1
u_OH = 1.8e-4 # cm^2 V^-1 s^-1
u_Na = 4.5e-5 # cm^2 V^-1 s^-1
u_Cl = 6.8e-5 # cm^2 V^-1 s^-1

H_0 = 1e-7 # adim
OH_0 = 1e-7 # adim
Na_0 = 1.0 # adim
Cl_0 = 1.0 # adim
H2O_0 = 347 # adim
C_0 = 1e8 # M
t0 = 1.0 # s
x0 = 1.0 # cm
Phi_0 = 5.0 # V

T = 298.0 # K
k_wf = 2.4917e8 # m^3 mol^-1 s^-1 revisar
k_wb = 2.7e-5 # s^-1  revisar

#F = 96485.3415 # A s mol^-1
#R = 8.31 # kg m^2 K^-1 mol^-1 s^-2
#σ_0 = 0.2 # S m^-1

t_max = 0.01 # adim
x_max = 1.0  # adim 

H_anode_rate = 4.2 # mol m^-3 s^-1
OH_cathode_rate = 5.0 # mol m^-3 s^-1
Na_anode_rate = 0.0 # adim
Na_cathode_rate = 2.0 * Na_0 # adim
Cl_anode_rate = 1.37 * Cl_0 # adim
Cl_cathode_rate = 0.0 # adim

Pe_Na = x0^2 / (t0 * D_Na)
M_Na = x0^2 / (t0 * Phi_0 * u_Na)
Pe_Cl = x0^2 / (t0 * D_Cl)
M_Cl = x0^2 / (t0 * Phi_0 * u_Cl)


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
        ( Dt(Na(t,x)) ~ (1.0/Pe_Na) * Dxx(Na(t,x)) 
                      + z_Na / (abs(z_Na)*M_Na) * Dx(Na(t,x)) * Phi_0 * (x_max-x) / x_max),
        ( Dt(Cl(t,x)) ~ (1.0/Pe_Cl) * Dxx(Cl(t,x)) 
                      + z_Cl / (abs(z_Cl)*M_Cl) * Dx(Cl(t,x)) * Phi_0 * (x_max-x) / x_max)
      ]

bcs = [ 
        Na(0.0,x) ~ Na_0,
        Cl(0.0,x) ~ Cl_0,
        Na(t,0.0) ~ Na_anode_rate,
        Cl(t,0.0) ~ Cl_anode_rate,
        Na(t,x_max) ~ Na_cathode_rate,
        Cl(t,x_max) ~ Cl_cathode_rate
      ]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,t_max),
           x ∈ IntervalDomain(0.0,x_max)]

# Discretization
dt = 5e-9
dx = 1e-2

# Neural network
dim = length(domains)
output = length(eqs)
neurons = 16
chain1 = FastChain( FastDense(dim,neurons,Flux.σ),
                    FastDense(neurons,neurons,Flux.σ),
                    FastDense(neurons,neurons,Flux.σ),
                    FastDense(neurons,1))
chain2 = FastChain( FastDense(dim,neurons,Flux.σ),
                    FastDense(neurons,neurons,Flux.σ),
                    FastDense(neurons,neurons,Flux.σ),
                    FastDense(neurons,1))

#strategy = GridTraining(dx=[dt,dx])
strategy = GridTraining(dx=dx)
#strategy = QuadratureTraining(algorithm=HCubatureJL(),reltol= 1e-5,abstol=1e-5,maxiters=30000)
#strategy = StochasticTraining(number_of_points = 100)

discretization = PhysicsInformedNN([chain1,chain2],strategy=strategy)

pde_system = PDESystem(eqs,bcs,domains,[t,x],[Na,Cl])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob,Optim.BFGS();cb=cb,maxiters=30000)
#res = GalacticOptim.solve(prob, ADAM(0.001), cb = cb, maxiters=5000)

phi = discretization.phi

initθ = discretization.initθ
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers = [res.minimizer[s] for s in sep]

ts,xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
ts_ = ts[1:10:size(ts)[1]]
Na_predict = [ [ phi[1]([t,x],minimizers[1])[1] for x in xs] for t in ts_] 
Cl_predict = [ [ phi[2]([t,x],minimizers[2])[1] for x in xs] for t in ts_] 

using Plots, LaTeXStrings
p1 = plot(xs, Na_predict, title = L"$Na^+ | mol \  m^{-3}$");
p2 = plot(xs, Cl_predict, title = L"$Cl^- | mol \ m^{-3}$");
plot(p1,p2)
savefig("1d-laplace-nerst-planck-Na-Cl-adim-neuralpde.svg")


