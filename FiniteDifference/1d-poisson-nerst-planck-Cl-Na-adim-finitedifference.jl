################################################################################
#   This code solves a simplified 1D Poisson-Nernst-Planck system using NeuralPDE
#
#   Equations:
#        d2Phi/dx2 =    ( 1.0 / Po_1 ) * ( z_Na^+ * Na^+ + z_Cl^- * Cl^- )
#        dNa^+/dt =     ( 1.0 / Pe_Na^+ ) * d2Na^+/dx2 
#                     + z_Na^+ / ( abs(z_Na^+) * M_Na^+ ) *
#                     ( dNa^+/dx * dPhi/dx + Na^+ * d2Phi/dx2 )
#        dCl^-/dt =     ( 1.0 / Pe_Cl^- ) * d2Cl^-/dx2
#                     + z_Cl^- / ( abs(z_Cl^-) * M_Cl^- ) 
#                     * ( dCl^-/dx * dPhi/dx + Cl^- * d2Phi/dx2 )
#
#   Initial conditions:
#        Phi(0,x) = 0.0
#        Na^+(0,x) = Na^+_0
#        Cl^-(0,x) = Cl^-_0
#
#   Boundary conditions (simplified):
#        Phi(t,0) = Phi_0
#        Phi(t,n) = 0.0
#        Na^+(t,0) = 0.0
#        Na^+(t,n) = 2.0 * Na^+_0
#        Cl^-(t,0) = 1.37 * Cl^-_0
#        Cl^-(t,n) = 0.0
#
#   How to run:
#        $ julia 1d-poisson-nerst-planck-Cl-Na-adim-finitedifference
#
################################################################################


# Solve a time step ############################################################
# It is used to avoid copying independent variables. See main iteration below.
function solve_time_step(Na,Cl,Phi,Na_aux,Cl_aux,Phi_aux)

    # Solve Nernst-Planck equation
    for i = 2:n-1
        Na[i] = (   (1.0/Pe_Na) * (Na_aux[i+1]-2.0*Na_aux[i]+Na_aux[i-1])/(dx^2)
                  + c_Na * (Na_aux[i+1]-Na_aux[i])/dx * (Phi[i+1]-Phi[i-1])/(2.0*dx)
                  + c_Na *  Na_aux[i] * (Phi[i+1]-2.0*Phi[i]+Phi[i-1])/(dx^2)
                ) * dt + Na_aux[i]
        Cl[i] = (   (1.0/Pe_Cl) * (Cl_aux[i+1]-2.0*Cl_aux[i]+Cl_aux[i-1])/(dx^2)
                  + c_Cl * (Cl_aux[i]-Cl_aux[i-1])/dx * (Phi[i+1]-Phi[i-1])/(2.0*dx)
                  + c_Cl * Cl_aux[i] * (Phi[i+1]-2.0*Phi[i]+Phi[i-1])/(dx^2)
                ) * dt + Cl_aux[i]
    end
#    Na[1] = Na_0 - Na_0 * it * dt / 10.0
#    Na[n] = Na_0 * it * dt / 10.0 + Na_0
#    Cl[1] = 0.37 * Cl_0 * it * dt / 10.0 + Cl_0
#    Cl[n] = Cl_0 - Cl_0 * it * dt / 10.0

    Na[1] = 0.0
    Na[n] = 2.0 * Na_0
    Cl[1] = 1.37 * Cl_0
    Cl[n] = 0.0

    # Solve Poisson equation
    e = 1.0
    while e > 0.0001
        for i = 2:n-1
            Phi_aux[i]  = ( Phi[i+1] + Phi[i-1] ) / 2.0 
                        + dx^2 / ( 2.0 * Po ) * ( z_Na * Na_aux[i] + z_Cl * Cl_aux[i] )
        end
        for i = 2:n-1
            Phi[i]  = ( Phi_aux[i+1] + Phi_aux[i-1] ) / 2.0 
                    + dx^2 / ( 2.0 * Po ) * ( z_Na * Na_aux[i] + z_Cl * Cl_aux[i] )
        end
        e = maximum(abs.(Phi_aux-Phi))
    end

    # Save results
    if it % it_save == 0 
        push!(Phis,copy(Phi))
        push!(Nas,copy(Na))
        push!(Cls,copy(Cl))
    end

end


# Parameters ###################################################################


# Electrochemical parameters

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

D_Na = 0.89e-7 # dm^2 s^−1
D_Cl = 1.36e-7 # dm^2 s^−1

u_Na = D_Na * abs(z_Na) * F / (R * T)
u_Cl = D_Cl * abs(z_Cl) * F / (R * T)

t_max = 0.01 / t_ref    # non-dim
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


# Delta t [s]
dt = 1e-5 # s

# Time iterations
time_it = floor(Int,t_max/dt)

# No. of spatial domain nodes
n = 100 #floor(Int,x_max/dx)

# Delta x
dx = x_max / ( n - 1.0 ) # m

# No. of iterations to print results
it_save = floor(time_it/4)  #floor(1.0/dt/20.0)


# Set initial and boundary conditions ##########################################

Na = zeros(n)
Cl = zeros(n)
Phi = zeros(n)
Na_aux = zeros(n)
Cl_aux = zeros(n)
Phi_aux = zeros(n)
Na .= Na_0
Cl .= Cl_0
Phi[1] = Phi_0
Phi_aux[1] = Phi_0
Na_aux .= Na_0
Cl_aux .= Cl_0

Phis = []
Nas = []
Cls = []
its = []

push!(Phis,Phi)
push!(Nas,Na)
push!(Cls,Cl)

# Solve equation system ########################################################
println("No. of time steps: ", time_it)
it = 1
while it < time_it
    # On each iteration two time steps are computed to avoid copying variables
    solve_time_step(Na_aux,Cl_aux,Phi_aux,Na,Cl,Phi)
    global it += 1
    solve_time_step(Na,Cl,Phi,Na_aux,Cl_aux,Phi_aux)
    global it += 1
    print("Progress: $(round(it/time_it*100.0; digits=2)) % \r")
    flush(stdout)
end

# Plot results #################################################################

using Plots, LaTeXStrings

dt = t_max / 4.0
ts = 0.0:dt:t_max
xs = 0.0:dx:x_max

labels = permutedims([ "$(t*t_ref) s" for t in ts ])
p1 = plot(xs * x_ref, Phis * Phi_ref,
          xlabel = "dm", ylabel = "V",title = L"$\Phi$",labels=labels);
savefig("Phi.svg")
p2 = plot(xs * x_ref, Nas * C_ref, 
          xlabel = "dm", ylabel = "M",title = L"$Na^+$",labels=labels);
savefig("Na.svg")
p3 = plot(xs * x_ref, Cls * C_ref,
          xlabel = "dm", ylabel = "M",title = L"$Cl^-$",labels=labels);
savefig("Cl.svg")


