using Flux, DiffEqFlux, OrdinaryDiffEq, Optim, Test, Plots

# Parameters ###################################################################

t_ref = 1.0       # s
x_ref = 0.38       # dm 
C_ref = 0.16      # mol/dm^3
Phi_ref = 1.0     # V

epsilon = 78.5    # K
F = 96485.3415    # A s mol^-1
R = 831.0         # kg dm^2 s^-2 K^-1 mol^-1 
T = 298.0         # K

z_Na = 1.0        # non-dim
z_Cl = -1.0       # non-dim

D_Na = 0.89e-5 #0.89e-7 # dm^2 s^−1
D_Cl = 1.36e-5 #1.36e-7 # dm^2 s^−1

u_Na = D_Na * abs(z_Na) * F / (R * T)
u_Cl = D_Cl * abs(z_Cl) * F / (R * T)

t_max = 1.0 / t_ref       # non-dim
x_max = 0.38 / x_ref      # non-dim
Na_0 = 0.16 / C_ref       # non-dim
Cl_0 = 0.16 / C_ref       # non-dim
Phi_0 = 4.0 / Phi_ref     # non-dim

Phi_anode = Phi_0         # non-dim
Phi_cathode = 0.0         # non-dim
Na_anode = 0.0            # non-dim
Na_cathode = 2.0 * Na_0   # non-dim
Cl_anode = 1.37 * Cl_0    # non-dim
Cl_cathode = 0.0          # non-dim

Pe_Na = x_ref^2 / ( t_ref * D_Na )  # non-dim
Pe_Cl = x_ref^2 / ( t_ref * D_Cl )  # non-dim

M_Na = x_ref^2 / ( t_ref * Phi_ref * u_Na )  # non-dim
M_Cl = x_ref^2 / ( t_ref * Phi_ref * u_Cl )  # non-dim

Po = (epsilon * Phi_ref) / (F * x_ref * C_ref)  # non-dim

c_Na = z_Na / ( abs(z_Na) * M_Na )  # non-dim

c_Cl = z_Cl / ( abs(z_Cl) * M_Cl )  # non-dim

# Numerical parameters #########################################################
n = 50 # No. of spatial domain nodes
dx = x_max / ( n - 1.0 ) # Delta x, m
datasize = 30
tspan = (0.0, t_max)
tsteps = range(tspan[1], tspan[2], length = datasize)

# Initial conditions ###########################################################
u0 = Array{Float64}(undef,3*n)
du0 = Array{Float64}(undef,3*n)
differential_vars = Array{Bool}(undef,3*n)
out = Array{Float64}(undef,n)
 
u0[1] = Na_anode
u0[2:n-1] .= 1.0
u0[n] = Na_cathode
differential_vars[1:n] .= true

u0[n+1] = Cl_anode
u0[n+2:2*n-1] .= 1.0
u0[2*n] = Cl_cathode
differential_vars[n+1:2*n] .= true

u0[2*n+1] = Phi_anode
u0[2*n+2:3*n-1] .= Phi_anode / 2.0
u0[3*n] = Phi_cathode
differential_vars[2*n+1:3*n] .= false

du0[1:n] .= (1.0/Pe_Na) * (-2.0)/(dx^2) + c_Na * (-1.0)/dx * Phi_0
du0[n+1:2*n] .= (1.0/Pe_Cl) * (-2.0)/(dx^2) + c_Cl * (1.0)/dx * Phi_0
du0[2*n+1:3*n] .= (-2.0) / (dx^2)

# Equation system discretized using MOL ########################################

function f!(du, u, p, t)

    du[1] = Na_anode - u[1]
    du[n] = Na_cathode - u[n]
    for i in 2:n-1
        du[i] =   (  (1.0/Pe_Na) * (u[i+1]-2.0*u[i]+u[i-1])/(dx^2)
                      + c_Na * (u[i+1]-u[i])/dx * (u[i+1+2*n]-u[i-1+2*n])/(2.0*dx)
                      + c_Na *  u[i] * (u[i+1+2*n]-2.0*u[i+2*n]+u[i-1+2*n])/(dx^2)
                   )
    end

    du[n+1] = Cl_anode - u[n+1]
    du[2*n] = Cl_cathode - u[2*n]
    for i in n+2:2*n-1
        du[i] =   (  (1.0/Pe_Cl) * (u[i+1]-2.0*u[i]+u[i-1])/(dx^2)
                      + c_Cl * (u[i]-u[i-1])/dx * (u[i+1+n]-u[i-1+n])/(2.0*dx)
                      + c_Cl *  u[i] * (u[i+1+n]-2.0*u[i+n]+u[i-1+n])/(dx^2)
                   )
    end

    du[2*n+1] = Phi_anode - u[2*n+1]
    du[3*n] = Phi_cathode - u[3*n]
    for i in 2*n+2:3*n-1
        du[i] =  ( u[i+1] - 2.0 * u[i] + u[i-1] ) / (dx^2)
                  - 1.0 / Po * ( z_Na * u[i-2*n] + z_Cl * u[i-n] )
    end

end


# Solution of MOL problem using ODE solver #####################################

M = zeros(3*n,3*n); for i in 1:2*n M[i,i]=1 end
M[1,1] = 0.0
M[n,n] = 0.0
M[n+1,n+1] = 0.0
M[2*n,2*n] = 0.0
M[2*n+1,2*n+1] = 0.0
M[3*n,3*n] = 0.0

stiff_func = ODEFunction(f!, mass_matrix = M)
prob_stiff = ODEProblem(stiff_func, u0, tspan)
sol_stiff = solve(prob_stiff, Rodas5(), saveat = 0.1)


plot(sol_stiff)

nn_dudt2 = FastChain(FastDense(3*n, 64, tanh),
                     FastDense(64, 2*n))
                     
                     
function g!(u,p,t)
    out[1] = Phi_anode - u[2*n+1]
    out[n] = Phi_cathode - u[3*n]
    for i in 2*n+2:3*n-1
        j = i - 2 * n
        out[j] =  ( u[i+1] - 2.0 * u[i] + u[i-1] ) / (dx^2)
                  - 1.0 / Po * ( z_Na * u[i-2*n] + z_Cl * u[i-n] )
    end
    out
end

model_stiff_ndae = NeuralODEMM(nn_dudt2, g!,
                               tspan, M, Rodas5(autodiff=false), saveat = 0.1)
model_stiff_ndae(u0)

function predict_stiff_ndae(p)
    return model_stiff_ndae(u0, p)
end

function loss_stiff_ndae(p)
    pred = predict_stiff_ndae(p)
    loss = sum(abs2, Array(sol_stiff) .- pred)
    return loss, pred
end

callback = function (p, l, pred) #callback function to observe training
  display(l)
  return false
end

l1 = first(loss_stiff_ndae(model_stiff_ndae.p))
result_stiff = DiffEqFlux.sciml_train(loss_stiff_ndae, model_stiff_ndae.p,
                                      BFGS(initial_stepnorm = 0.001),
                                      cb = callback, maxiters = 200)

