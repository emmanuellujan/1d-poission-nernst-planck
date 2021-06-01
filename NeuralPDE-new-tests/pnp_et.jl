using NeuralPDE
using Quadrature, Cubature, Cuba
using Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
#using Plots
using DelimitedFiles
using QuasiMonteCarlo
using JLD

# Parameters 

t_ref = 1.0       # s
x_ref = 0.38      # dm 
C_ref = 0.16      # mol/dm^3
Phi_ref = 1.0     # V

epsilon = 78.5    # K
F = 96485.3415    # A s mol^-1
R = 831.0         # kg dm^2 s^-2 K^-1 mol^-1 
T = 298.0         # K

z_Na = 1.0        # non-dim
z_Cl = -1.0       # non-dim

D_Na = 0.89e-7    # dm^2 s^−1
D_Cl = 1.36e-7    # dm^2 s^−1

u_Na = D_Na * abs(z_Na) * F / (R * T)
u_Cl = D_Cl * abs(z_Cl) * F / (R * T)

t_max = 0.01 / t_ref    # non-dim
x_max = 0.38 / x_ref    # non-dim
Na_0 = 0.16 / C_ref     # non-dim
Cl_0 = 0.16 / C_ref     # non-dim
Phi_0 = 4.0 / Phi_ref   # non-dim

Na_anode = 0.0            # non-dim
Na_cathode = 2.0 * Na_0   # non-dim
Cl_anode = 1.37 * Cl_0    # non-dim
Cl_cathode = 0.0          # non-dim

Pe_Na = x_ref^2 / ( t_ref * D_Na )  # non-dim
Pe_Cl = x_ref^2 / ( t_ref * D_Cl )  # non-dim

M_Na = x_ref^2 / ( t_ref * Phi_ref * u_Na )  # non-dim
M_Cl = x_ref^2 / ( t_ref * Phi_ref * u_Cl )  # non-dim

Po_1 = (epsilon * Phi_ref) / (F * x_ref * C_ref)  # non-dim

dx = 0.01 # non-dim
dt = 0.0001 # non-dim


function pnp(strategy, minimizer, maxIters)

    # Parameters and variables
    @parameters x, t
    @variables Phi(..), Na(..), Cl(..)
    Dt = Differential(t)
    Dx = Differential(x)
    Dxx = Differential(x)^2

    # Equations, initial and boundary conditions ###############################

    eqs = [
            ( Dxx(Phi(x, t)) ~ ( 1.0 / Po_1 ) *
                              ( z_Na * Na(x, t) + z_Cl * Cl(x, t) ) )
            ,
            ( Dt(Na(x, t)) ~ ( 1.0 / Pe_Na ) * Dxx(Na(x, t)) 
                          +   z_Na / ( abs(z_Na) * M_Na ) 
                          * ( Dx(Na(x, t)) * Dx(Phi(x, t)) + Na(x, t) * Dxx(Phi(x, t)) ) )
            ,
            ( Dt(Cl(x, t)) ~ ( 1.0 / Pe_Cl ) * Dxx(Cl(x, t)) 
                          +   z_Cl / ( abs(z_Cl) * M_Cl ) 
                          * ( Dx(Cl(x, t)) * Dx(Phi(x, t)) + Cl(x, t) * Dxx(Phi(x, t)) ) )
          ]

    bcs = [
            Phi(0.0, t) ~ Phi_0,
            Phi(x_max, t) ~ 0.0
            ,
            Na(x, 0.0) ~ Na_0,
            Na(0.0, t) ~ Na_anode,
            Na(x_max, t) ~ Na_cathode
            ,
            Cl(x, 0.0) ~ Cl_0,
            Cl(0.0, t) ~ Cl_anode,
            Cl(x_max, t) ~ Cl_cathode
          ]

    # Space and time domains ###################################################

    domains = [
                x ∈ IntervalDomain(0.0, x_max),
                t ∈ IntervalDomain(0.0, t_max)
              ]

    xs,ts = [domain.domain.lower:dx/10:domain.domain.upper for (dx,domain) in zip([dx,dt],domains)]

    indvars = [x,t]
    depvars = [Phi,Na,Cl]

    chain_1 = FastChain(FastDense(2,18,Flux.σ),FastDense(18,18,Flux.σ),FastDense(18,1))
    chain_2 = FastChain(FastDense(2,18,Flux.σ),FastDense(18,18,Flux.σ),FastDense(18,1))
    chain_3 = FastChain(FastDense(2,18,Flux.σ),FastDense(18,18,Flux.σ),FastDense(18,1))

    losses = []
    error = []
    times = []
    
    dx_err = x_max/10

    error_strategy = GridTraining(dx_err)

    phi_1 = NeuralPDE.get_phi(chain_1)
    phi_2 = NeuralPDE.get_phi(chain_2)
    phi_3 = NeuralPDE.get_phi(chain_3)
    derivative = NeuralPDE.get_numeric_derivative()
    initθ_1 = DiffEqFlux.initial_params(chain_1)
    initθ_2 = DiffEqFlux.initial_params(chain_2)
    initθ_3 = DiffEqFlux.initial_params(chain_3)

    _pde_loss_function_1 = NeuralPDE.build_loss_function(eqs[1],indvars,depvars,phi_1,
                           derivative,chain_1,initθ_1,error_strategy)
    _pde_loss_function_2 = NeuralPDE.build_loss_function(eqs[2],indvars,depvars,phi_2,
                           derivative,chain_2,initθ_2,error_strategy)
    _pde_loss_function_3 = NeuralPDE.build_loss_function(eqs[3],indvars,depvars,phi_3,
                           derivative,chain_3,initθ_3,error_strategy)

    bc_indvars_1 = NeuralPDE.get_variables(bcs[1:2],indvars,depvars)
    bc_indvars_2 = NeuralPDE.get_variables(bcs[3:5],indvars,depvars)
    bc_indvars_3 = NeuralPDE.get_variables(bcs[6:8],indvars,depvars)
    
    _bc_loss_functions_1 = [NeuralPDE.build_loss_function(bc,indvars,depvars,phi_1,
                            derivative,chain_1,initθ_1,error_strategy,
                            bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs[1:2],bc_indvars_1)]
                            
    _bc_loss_functions_2 = [NeuralPDE.build_loss_function(bc,indvars,depvars,phi_2,
                            derivative,chain_2,initθ_2,error_strategy,
                            bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs[3:5],bc_indvars_2)]
                            
    _bc_loss_functions_3 = [NeuralPDE.build_loss_function(bc,indvars,depvars,phi_3,
                            derivative,chain_3,initθ_3,error_strategy,
                            bc_indvars = bc_indvar) for (bc,bc_indvar) in zip(bcs[6:8],bc_indvars_3)]

    train_sets_1 = NeuralPDE.generate_training_sets(domains,[x_max/10,t_max/10],[eqs[1]],bcs[1:2],indvars,depvars)
    train_domain_set_1, train_bound_set_1 = train_sets_1
    
    train_sets_2 = NeuralPDE.generate_training_sets(domains,[x_max/10,t_max/10],[eqs[2]],bcs[3:5],indvars,depvars)
    train_domain_set_2, train_bound_set_2 = train_sets_2

    train_sets_3 = NeuralPDE.generate_training_sets(domains,[x_max/10,t_max/10],[eqs[3]],bcs[6:8],indvars,depvars)
    train_domain_set_3, train_bound_set_3 = train_sets_3

    pde_loss_function_1 = NeuralPDE.get_loss_function([_pde_loss_function_1],
                                                        train_domain_set_1,
                                                        error_strategy)
                                                        
    pde_loss_function_2 = NeuralPDE.get_loss_function([_pde_loss_function_2],
                                                        train_domain_set_2,
                                                        error_strategy)
                                                        
    pde_loss_function_3 = NeuralPDE.get_loss_function([_pde_loss_function_3],
                                                        train_domain_set_3,
                                                        error_strategy)

    bc_loss_function_1 = NeuralPDE.get_loss_function(_bc_loss_functions_1,
                                                      train_bound_set_1,
                                                      error_strategy)
                                                      
    bc_loss_function_2 = NeuralPDE.get_loss_function(_bc_loss_functions_2,
                                                      train_bound_set_2,
                                                      error_strategy)
                                                      
    bc_loss_function_3 = NeuralPDE.get_loss_function(_bc_loss_functions_3,
                                                      train_bound_set_3,
                                                      error_strategy)


    function loss_function_(θ,p)
        return pde_loss_function_1(θ) + bc_loss_function_1(θ) +
               pde_loss_function_2(θ) + bc_loss_function_2(θ) +
               pde_loss_function_3(θ) + bc_loss_function_3(θ)
    end

    cb_ = function (p,l)
        deltaT_s = time_ns() #Start a clock when the callback begins, this will evaluate questo misurerà anche il calcolo degli uniform error

        ctime = time_ns() - startTime - timeCounter #This variable is the time to use for the time benchmark plot
        append!(times, ctime/10^9) #Conversion nanosec to seconds
        append!(losses, l)
        #append!(error, pde_loss_function(p) + bc_loss_function(p))
        append!(error,  pde_loss_function_1(p) + bc_loss_function_1(p) +
                        pde_loss_function_2(p) + bc_loss_function_2(p) +
                        pde_loss_function_3(p) + bc_loss_function_3(p))
        
        #println(length(losses), " Current loss is: ", l, " uniform error is, ",  pde_loss_function(p) + bc_loss_function(p))
        println(length(losses), " Current loss is: ", l, " uniform error is, ",  
                pde_loss_function_1(p) + bc_loss_function_1(p) +
                pde_loss_function_2(p) + bc_loss_function_2(p) +
                pde_loss_function_3(p) + bc_loss_function_3(p))

        timeCounter = timeCounter + time_ns() - deltaT_s #timeCounter sums all delays due to the callback functions of the previous iterations

        return false
    end

    discretization = PhysicsInformedNN([chain_1,chain_2,chain_3],strategy)

    pde_system = PDESystem(eqs,bcs,domains,indvars,depvars)
    prob = discretize(pde_system,discretization)

    timeCounter = 0.0
    startTime = time_ns() #Fix initial time (t=0) before starting the training

    res = GalacticOptim.solve(prob, minimizer; cb = cb_, maxiters = maxIters)
    phi = discretization.phi

    params = res.minimizer

    # Model prediction
    domain = [x,t]

    Phi_predict = reshape([first(phi[1]([x,t],res.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))
    Na_predict = reshape([first(phi[2]([x,t],res.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))
    Cl_predict = reshape([first(phi[3]([x,t],res.minimizer)) for x in xs for t in ts],(length(xs),length(ts)))

    return [error, params, domain, times, [Phi_predict, Na_predict, Cl_predict], losses]
end
