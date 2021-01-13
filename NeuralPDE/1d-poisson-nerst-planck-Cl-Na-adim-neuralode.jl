using DiffEqFlux, OrdinaryDiffEq, Flux, Optim, Plots

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

t_max = 1.0 / t_ref    # non-dim
x_max = 0.38 / x_ref    # non-dim
Na_0 = 0.16 / C_ref     # non-dim
Cl_0 = 0.16 / C_ref     # non-dim
Phi_0 = 4.0 / Phi_ref  # non-dim

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
u0 = Array{Float32}(undef,2*n) 
u0 .= 1.0
u0[1] = Na_anode
u0[n] = Na_cathode
u0[n+1] = Cl_anode
u0[2*n] = Cl_cathode

# Equation system discretized using MOL ########################################
function trueODEfunc(du, u, p, t)

    u0[1] = Na_anode
    u0[n] = Na_cathode
    for i in 2:n
        du[i] =    (  (1.0/Pe_Na) * (u[i+1]-2.0*u[i]+u[i-1])/(dx^2)
                      + c_Na * (u[i+1]-u[i])/dx * Phi_0
                   )
    end

    u0[n+1] = Cl_anode
    u0[2*n] = Cl_cathode
    for i in n+2:2*n-1
        du[i] =    (  (1.0/Pe_Cl) * (u[i+1]-2.0*u[i]+u[i-1])/(dx^2)
                      + c_Cl * (u[i]-u[i-1])/dx * Phi_0
                   )
    end

end

# Solution of MOL problem using ODE solver #####################################
prob_trueode = ODEProblem(trueODEfunc, u0, tspan)
ode_data = Array(solve(prob_trueode, Tsit5(), saveat = tsteps))

# Solution of MOL problem using PINN/NeuralODE #################################
chain = FastChain((x,p)-> x,
                  FastDense(100, 32, tanh),
                  FastDense(32, 100))
prob_neuralode = NeuralODE(chain, tspan, Vern7(), saveat = tsteps, abstol=1e-6, reltol=1e-6)

function predict_neuralode(p)
  Array(prob_neuralode(u0, p))
end

function loss_neuralode(p)
    pred = predict_neuralode(p)
    loss = sum(abs2, (ode_data[:,1:size(pred,2)] .- pred))
    return loss, pred
end

iter = 0
callback = function (p, l, pred; doplot = true)
  global iter
  iter += 1

  display(l)
  if doplot
    # plot current prediction against data
    p1 = scatter(ode_data[1:50,datasize], label = "data")
    scatter!(pred[1:50,datasize], label = "prediction")
    p2 = scatter(ode_data[51:100,datasize], label = "data")
    scatter!(pred[51:100,datasize], label = "prediction")
    display(plot(p1,p2))
  end

  return false
end

result_neuralode = DiffEqFlux.sciml_train(loss_neuralode, prob_neuralode.p,
                                           ADAM(0.05), cb = callback,
                                           maxiters = 300)

# Plot and save result #########################################################

callback(result_neuralode.minimizer,loss_neuralode(result_neuralode.minimizer)...;doplot=true)
savefig("plot.png")
