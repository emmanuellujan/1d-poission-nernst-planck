################################################################################
#   This code solves a simplified 1D Poisson-Nernst-Planck system using NeuralPDE
#
#   Equations:
#        d2Phi/dx2 =    ( 1.0 / Po_1 ) * ( z_Na^+ * Na^+ + z_Cl^- * Cl^- )
#                     + ( 1.0 / Po_2 ) * ( z_H^+ * H^+ + z_OH^- * OH^- )
#        dNa^+/dt =     ( 1.0 / Pe_Na^+ ) * d2Na^+/dx2 
#                     + z_Na^+ / ( abs(z_Na^+) * M_Na^+ ) *
#                     ( dNa^+/dx * dPhi/dx + Na^+ * d2Phi/dx2 )
#        dCl^-/dt =     ( 1.0 / Pe_Cl^- ) * d2Cl^-/dx2
#                     + z_Cl^- / ( abs(z_Cl^-) * M_Cl^- ) 
#                     * ( dCl^-/dx * dPhi/dx + Cl^- * d2Phi/dx2 )
#        dH^+/dt =      ( 1.0 / Pe_H^+ ) * d2H^+/dx2 
#                     + z_H^+ / ( abs(z_H^+) * M_H^+ )
#                     * ( dH^+/dx * dPhi/dx + H^+ * d2Phi/dx2 )
#                     + k_wb * H2O - k_wf * H^+ * OH^-
#        dOH^-/dt =     ( 1.0 / Pe_OH^- ) * d2OH^-/dx2 
#                     + z_OH^- / ( abs(z_OH^-))*M_OH^- )
#                     * ( dOH^-/dx * dPhi/dx + OH^- * d2Phi/dx2 )
#                     + k_wb * H2O - k_wf * H^+ * OH^-
#
#   Initial conditions:
#        Phi(0,x) = 0.0
#        Na^+(0,x) = Na^+_0
#        Cl^-(0,x) = Cl^-_0
#        H^+(0,x) = H^+_0
#        OH^-(0,x) = OH^-_0
#
#   Boundary conditions (simplified):
#        Phi(t,0) = Phi_0
#        Phi(t,n) = 0.0
#        Na^+(t,0) = 0.0
#        Na^+(t,n) = 2.0 * Na^+_0
#        Cl^-(t,0) = 1.37 * Cl^-_0
#        Cl^-(t,n) = 0.0
#        H^+(t,0) = 1.25 * H^+_0
#        H^+(t,n) = H^+_0
#        OH^-(t,0) = OH^-_0
#        OH^-(t,n) = 1.25 * OH^-_0
#
#   How to run:
#        julia 1d-poisson-nernst-planck-nondim-neuralpde.jl
#
#   TODO: include H^+ and OH^-
#
#
#################################################################################

using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
using Quadrature, Cubature, Cuba

# Parameters ###################################################################

@parameters t,x
@variables Phi(..)
@variables Na(..),Cl(..)
#@variables H(..),OH(..)
@derivatives Dt'~t
@derivatives Dx'~x
@derivatives Dxx''~x

t_ref = 1.0       # s
x_ref = 0.4       # dm 
C_ref_1 = 0.16    # mol/dm^3
C_ref_2 = 1e-7    # mol/dm^3
Phi_ref = 1.0     # V

epsilon = 78.5    # K
F = 96485.3415    # A s mol^-1
R = 8.31          # kg m^2 K^-1 mol^-1 s^-2
T = 298.0         # K

z_Na = 1.0        # non-dim
z_Cl = -1.0       # non-dim
z_H = 1.0         # non-dim
z_OH = -1.0       # non-dim

D_Na = 0.89e-7    # dm^2 s^−1
D_Cl = 1.36e-7    # dm^2 s^−1
D_H = 6.25e-7     # dm^2 s^-1
D_OH =3.52e-7     # dm^2 s^−1

u_Na = D_Na * abs(z_Na) * F / (R * T)
u_Cl = D_Cl * abs(z_Cl) * F / (R * T)
u_H = D_H * abs(z_H) * F / (R * T)
u_OH = D_OH * abs(z_OH) * F / (R * T)

t_max = 0.01 / t_ref    # non-dim
x_max = 0.4 / x_ref     # non-dim
Na_0 = 0.16 / C_ref_1   # non-dim
Cl_0 = 0.16 / C_ref_1   # non-dim
H_0 = 1e-7 / C_ref_2    # non-dim
OH_0 = 1e-7 / C_ref_2   # non-dim
H2O_0 = 55.5 / C_ref_2  # non-dim
Phi_0 = 4.0 / Phi_ref   # non-dim

k_wf = 1.5e11           # dm^3 mol^-1 s^-1
k_wb = 2.7e-5           # s^-1 

Na_anode = 0.0            # non-dim
Na_cathode = 2.0 * Na_0   # non-dim
Cl_anode = 1.37 * Cl_0    # non-dim
Cl_cathode = 0.0          # non-dim
H_anode = 1.25 * H_0      # non-dim
H_cathode = H_0           # non-dim
OH_anode = OH_0           # non-dim
OH_cathode = 1.25 * OH_0  # non-dim

Pe_Na = x_ref^2 / ( t_ref * D_Na )  # non-dim
Pe_Cl = x_ref^2 / ( t_ref * D_Cl )  # non-dim
Pe_H = x_ref^2 / ( t_ref * D_H )    # non-dim
Pe_OH = x_ref^2 / ( t_ref * D_OH )  # non-dim

M_Na = x_ref^2 / ( t_ref * Phi_ref * u_Na )  # non-dim
M_Cl = x_ref^2 / ( t_ref * Phi_ref * u_Cl )  # non-dim
M_H = x_ref^2 / ( t_ref * Phi_ref * u_H )    # non-dim
M_OH = x_ref^2 / ( t_ref * Phi_ref * u_OH )  # non-dim

Po_1 = (epsilon * Phi_ref) / (F * x_ref * C_ref_1)  # non-dim
Po_2 = (epsilon * Phi_ref) / (F * x_ref * C_ref_2)  # non-dim


# Equations, initial and boundary conditions ###################################

eqs = [
#        ( Dxx(Phi(t,x)) ~ ( 1.0 / Po_1 ) * ( z_Na * Na(t,x) + z_Cl * Cl(t,x) )
#                          +( 1.0 / Po_2 ) * ( z_Na * H(t,x)  + z_Cl * OH(t,x) ) )
#        ,
        ( Dxx(Phi(t,x)) ~ ( 1.0 / Po_1 ) * ( z_Na * Na(t,x) + z_Cl * Cl(t,x) ) )
        ,
        ( Dt(Na(t,x)) ~ ( 1.0 / Pe_Na ) * Dxx(Na(t,x)) 
                      +   z_Na / ( abs(z_Na) * M_Na ) 
                      * ( Dx(Na(t,x)) * Dx(Phi(t,x)) + Na(t,x) * Dxx(Phi(t,x)) ) )
        ,
        ( Dt(Cl(t,x)) ~ ( 1.0 / Pe_Cl ) * Dxx(Cl(t,x)) 
                      +   z_Cl / ( abs(z_Cl) * M_Cl ) 
                      * ( Dx(Cl(t,x)) * Dx(Phi(t,x)) + Cl(t,x) * Dxx(Phi(t,x)) ) )
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
        Phi(t,0.0) ~ Phi_0,
        Phi(t,x_max) ~ 0.0
        ,
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

# Space and time domains #######################################################
domains = [t ∈ IntervalDomain(0.0,t_max),
           x ∈ IntervalDomain(0.0,x_max)]

# Neural network, Discretization ###############################################
dx = 1e-2 #, dt = 5e-9
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
chain3 = FastChain( FastDense(dim,neurons,Flux.σ),
                    FastDense(neurons,neurons,Flux.σ),
                    FastDense(neurons,neurons,Flux.σ),
                    FastDense(neurons,1))
#chain4 = FastChain( FastDense(dim,neurons,Flux.σ),
#                    FastDense(neurons,neurons,Flux.σ),
#                    FastDense(neurons,neurons,Flux.σ),
#                    FastDense(neurons,1))
#chain5 = FastChain( FastDense(dim,neurons,Flux.σ),
#                    FastDense(neurons,neurons,Flux.σ),
#                    FastDense(neurons,neurons,Flux.σ),
#                    FastDense(neurons,1))

strategy = GridTraining(dx=dx)
#strategy = GridTraining(dx=[dt,dx])
#strategy = QuadratureTraining(algorithm=HCubatureJL(),
#                              reltol= 1e-5,abstol=1e-5,maxiters=50000)
#strategy = StochasticTraining(number_of_points = 100)

discretization = PhysicsInformedNN([chain1,chain2,chain3],strategy=strategy)
#discretization = PhysicsInformedNN([chain1,chain2,chain3,chain4,chain5],
#                                   strategy=strategy)

pde_system = PDESystem(eqs,bcs,domains,[t,x],[Phi,Na,Cl])
#pde_system = PDESystem(eqs,bcs,domains,[t,x],[Phi,Na,Cl,H,OH])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob,Optim.BFGS();cb=cb,maxiters=30000)
#res = GalacticOptim.solve(prob, ADAM(0.001), cb = cb, maxiters=5000)


# Plots ########################################################################

phi = discretization.phi
initθ = discretization.initθ
acum =  [0;accumulate(+, length.(initθ))]
sep = [acum[i]+1 : acum[i+1] for i in 1:length(acum)-1]
minimizers = [res.minimizer[s] for s in sep]

ts,xs = [domain.domain.lower:dx:domain.domain.upper for domain in domains]
ts_ = ts[1:10:size(ts)[1]]
Phi_predict = [ [ phi[1]([t,x],minimizers[1])[1] for x in xs] for t in ts_] 
Na_predict = [ [ phi[2]([t,x],minimizers[2])[1] for x in xs] for t in ts_] 
Cl_predict = [ [ phi[3]([t,x],minimizers[3])[1] for x in xs] for t in ts_] 
#H_predict = [ [ phi[4]([t,x],minimizers[4])[1] for x in xs] for t in ts_] 
#OH_predict = [ [ phi[5]([t,x],minimizers[5])[1] for x in xs] for t in ts_] 

using Plots, LaTeXStrings
p1 = plot(xs * x_ref, Phi_predict * Phi_ref,
          xlabel = "cm", ylabel = "V",title = L"$\Phi$",
          label=["0.01 s"]);
savefig("Phi.svg")
p2 = plot(xs * x_ref, Na_predict * C_ref_1, 
          xlabel = "cm", ylabel = "M",title = L"$Na^+$",
          label=["0.01 s"]);
savefig("Na.svg")
p3 = plot(xs * x_ref, Cl_predict * C_ref_1,
          xlabel = "cm", ylabel = "M",title = L"$Cl^-$",
          label=["0.01 s"]);
savefig("Cl.svg")
#p4 = plot(xs, H_predict * C_ref_2,
#          xlabel = "cm", ylabel = "M",title = L"$H^+$",
#          label=["0.01 s"]);
#savefig("H.svg")
#p4 = plot(xs, OH_predict * C_ref_2,
#          xlabel = "cm", ylabel = "M",title = L"$OH^-$",
#          label=["0.01 s"]);
#savefig("OH.svg")


