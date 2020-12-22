#################################################################################
#   This code solves a simplified 1D Laplace-Nernst-Planck system using NeuralPDE
#
#   Equations:
#        dNa/dt =  ( 1.0 / Pe_Na ) * d2Na/dx2 
#                 + z_Na / ( abs(z_Na) * M_Na ) * ( dNa/dx * dPhi/dx + Na * d2Phi/dx2 )
#        dCl/dt = ( 1.0 / Pe_Cl ) * d2Cl/dx2
#                 + z_Cl / ( abs(z_Cl) * M_Cl ) * ( dCl/dx * dPhi/dx + Cl * d2Phi/dx2 )
#        dH/dt =  ( 1.0 / Pe_H ) * d2H/dx2 
#                 + z_H / ( abs(z_H) * M_H ) * ( dH/dx * dPhi/dx + H * d2Phi/dx2 )
#                 + k_wb * H2O - k_wf * H * OH
#        dOH/dt =  ( 1.0 / Pe_OH ) * d2OH/dx2 
#                 + z_H / ( abs(z_OH)*M_OH ) * ( dOH/dx * dPhi/dx + OH * d2Phi/dx2 )
#                 + k_wb * H2O - k_wf * H * OH
#        Phi = Phi_0 / x_max * x   (Laplace equation is solved analytically)
#
#   Initial conditions:
#        Na(0,x) = Na_0
#        Cl(0,x) = Cl_0
#        H(0,x) = H_0
#        OH(0,x) = OH_0
#
#   Boundary conditions (simplified):
#        Na(t,0) = 0.0
#        Na(t,n) = 2.0 * Na_0
#        Cl(t,0) = 1.37 * Cl_0
#        Cl(t,n) = 0.0
#        H(t,0) = 1.25 * H_0
#        H(t,n) = H_0
#        OH(t,0) = OH_0
#        OH(t,n) = 1.25 * OH_0
#
#   How to run:
#        julia 1d-laplace-nernst-planck-adim-neuralpde.jl
#
#################################################################################

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature, Cubature, Cuba

@parameters t,x
@variables Na(..),Cl(..)
#@variables H(..),OH(..)
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dxx''~x

# Parameters

t_ref = 1.0 # s
x_ref = 3.0 # cm
C_ref_1 = 1e-4 # M
C_ref_2 = 0.16 # M
Phi_ref = 1.0 # V

z_Na = 1.0
z_Cl = -1.0
z_H = 1.0
z_OH = -1.0

D_Na = 1.33e-5 # cm^2 s^−1
D_Cl = 2.03e-5 # cm^2 s^−1
D_H = 9.31e-5 # cm^2 s^-1
D_OH = 5.26e-5 # cm^2 s^−1 

u_Na = 4.5e-5 # cm^2 V^-1 s^-1
u_Cl = 6.8e-5 # cm^2 V^-1 s^-1
u_H = 3.24e-4 # cm^2 V^-1 s^-1
u_OH = 1.8e-4 # cm^2 V^-1 s^-1

t_max = 0.01 / t_ref # adim
x_max = 3.0 / x_ref # adim 
Na_0 = 0.16 / C_ref_2 # adim
Cl_0 = 0.16 / C_ref_2 # adim
H_0 = 1e-4 / C_ref_1 # adim
OH_0 = 1e-4 / C_ref_1 # adim
H2O_0 = 55.5 / C_ref_1 # adim
Phi_0 = 1.0 / Phi_ref # adim

k_wf = 2.4917e8 # m^3 mol^-1 s^-1 revisar
k_wb = 2.7e-5 # s^-1  revisar

Na_anode = 0.0 # adim
Na_cathode = 2.0 * Na_0 # adim
Cl_anode = 1.37 * Cl_0 # adim
Cl_cathode = 0.0 # adim
H_anode = 1.25 * H_0 # adim
H_cathode = H_0 # adim
OH_anode = OH_0 # adim
OH_cathode = 1.25 * OH_0 # adim

Pe_Na = x_ref^2 / ( t_ref * D_Na )
Pe_Cl = x_ref^2 / ( t_ref * D_Cl )
Pe_H = x_ref^2 / ( t_ref * D_H )
Pe_OH = x_ref^2 / ( t_ref * D_OH )

M_Na = x_ref^2 / ( t_ref * Phi_ref * u_Na )
M_Cl = x_ref^2 / ( t_ref * Phi_ref * u_Cl )
M_H = x_ref^2 / ( t_ref * Phi_ref * u_H )
M_OH = x_ref^2 / ( t_ref * Phi_ref * u_OH )


# Equations, initial and boundary conditions

eqs = [
        ( Dt(Na(t,x)) ~ ( 1.0 / Pe_Na ) * Dxx(Na(t,x)) 
                      + z_Na / ( abs(z_Na) * M_Na ) * Dx(Na(t,x)) * Phi_0 * ( x_max - x ) / x_max )
        ,
        ( Dt(Cl(t,x)) ~ ( 1.0 / Pe_Cl ) * Dxx(Cl(t,x)) 
                      + z_Cl / ( abs(z_Cl) * M_Cl ) * Dx(Cl(t,x) ) * Phi_0 * ( x_max - x ) / x_max )
#        ,
#        ( Dt(H(t,x)) ~ ( 1.0 / Pe_H ) * Dxx(H(t,x)) 
#                      + z_H / ( abs(z_H) * M_H ) * Dx(H(t,x)) * Phi_0 * ( x_max - x ) / x_max
#                      + k_wb * t_ref * H2O_0 - k_wf * C_ref_1 * t_ref * H(t,x) * OH(t,x) )
#        ,
#        ( Dt(OH(t,x)) ~ ( 1.0 / Pe_OH ) * Dxx(OH(t,x)) 
#                      + z_OH / ( abs(z_Cl) * M_OH ) * Dx(OH(t,x)) * Phi_0 * ( x_max - x ) / x_max
#                      + k_wb * t_ref * H2O_0 - k_wf * C_ref_1 * t_ref * H(t,x) * OH(t,x) )
      ]

bcs = [ 
        Na(0.0,x) ~ Na_0,
        Na(t,0.0) ~ Na_anode,
        Na(t,x_max) ~ Na_cathode
        ,
        Cl(0.0,x) ~ Cl_0,
        Cl(t,0.0) ~ Cl_anode,
        Cl(t,x_max) ~ Cl_cathode
#        ,
#        H(0.0,x) ~ H_0,
#        H(t,0.0) ~ H_anode,
#        H(t,x_max) ~ H_cathode
#        ,
#        OH(0.0,x) ~ OH_0,
#        OH(t,0.0) ~ OH_anode,
#        OH(t,x_max) ~ OH_cathode

      ]

# Space and time domains
domains = [t ∈ IntervalDomain(0.0,t_max),
           x ∈ IntervalDomain(0.0,x_max)]

# Discretization
dx = 1e-2 #, dt = 5e-9

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
#chain3 = FastChain( FastDense(dim,neurons,Flux.σ),
#                    FastDense(neurons,neurons,Flux.σ),
#                    FastDense(neurons,neurons,Flux.σ),
#                    FastDense(neurons,1))
#chain4 = FastChain( FastDense(dim,neurons,Flux.σ),
#                    FastDense(neurons,neurons,Flux.σ),
#                    FastDense(neurons,neurons,Flux.σ),
#                    FastDense(neurons,1))

strategy = GridTraining(dx=dx)
#strategy = GridTraining(dx=[dt,dx])
#strategy = QuadratureTraining(algorithm=HCubatureJL(),reltol= 1e-5,abstol=1e-5,maxiters=50000)
#strategy = StochasticTraining(number_of_points = 100)

#discretization = PhysicsInformedNN([chain1,chain2,chain3,chain4],strategy=strategy)
discretization = PhysicsInformedNN([chain1,chain2],strategy=strategy)

#pde_system = PDESystem(eqs,bcs,domains,[t,x],[H,OH,Na,Cl])
pde_system = PDESystem(eqs,bcs,domains,[t,x],[Na,Cl])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l, $p")
    return false
end

res = GalacticOptim.solve(prob,Optim.BFGS();cb=cb,maxiters=35000)
#res = GalacticOptim.solve(prob, ADAM(0.001), cb = cb, maxiters=5000)


# Plots

phi = discretization.phi
initθ = discretization.initθ
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers = [res.minimizer[s] for s in sep]

ts,xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
ts_ = ts[1:10:size(ts)[1]]
Na_predict = [ [ phi[1]([t,x],minimizers[1])[1] for x in xs] for t in ts_] 
Cl_predict = [ [ phi[2]([t,x],minimizers[2])[1] for x in xs] for t in ts_] 
#H_predict = [ [ phi[3]([t,x],minimizers[3])[1] for x in xs] for t in ts_] 
#OH_predict = [ [ phi[4]([t,x],minimizers[3])[1] for x in xs] for t in ts_] 

using Plots, LaTeXStrings
p1 = plot(xs * x_ref, Na_predict * C_ref_2, title = L"$Na^+ | M$");
p2 = plot(xs * x_ref, Cl_predict * C_ref_2, title = L"$Cl^- | M$");
#p3 = plot(xs, H_predict * C_ref_1, title = L"$H^+ | M$");
#p4 = plot(xs, OH_predict * C_ref_1, title = L"$OH^- | M$");
#plot(p1,p2,p3,p4)
plot(p1,p2)
savefig("1d-laplace-nerst-planck-adim-neuralpde.svg")


