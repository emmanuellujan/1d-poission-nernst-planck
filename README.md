# [WIP] These codes solve a Poisson-Nernst-Planck and Laplace-Nernst-Planck 1D system usign NeuralPDE and the Finite Difference method

The equation system is based on "pH front tracking in the electrochemical treatment (EChT) of tumors: Experiments and simulations", 
P. Turjanski, N. Olaiz, P. Abou-Adal, C. Suárez 1 , M. Risk 1 , G. Marshall". Electrochimica Acta 54 (2009) 6199–6206.

1D Poisson-Nernst-Planck and Laplace-Nernst-Planck

##   Equations
        d2φ/dx2 = -F/epsilon * ( z_H * H + z_OH * OH)   or  d2φ/dx2 = 0 
        dH/dt =  D_H * d2H/dx2 
                 + c_H * ( dH/dx * dφ/dx + H * d2φ/dx2 )
                 + k_wb * H2O - k_wf * H * OH
        dOH/dt = D_OH * d2OH/dx2
                 + c_OH * ( dOH/dx * dφ/dx + OH * d2φ/dx2 )
                 + k_wb * H2O - k_wf * H * OH

##   Initial conditions
        H(0,x) = H_0
        OH(0,x) = OH_0
        φ(0,x) = 0

##   Boundary conditions

Butler-Volmer equations have been replaced by a linear approximation.

        H(t,0) = H_anode_rate * t + H_0
        dH(t,n)/dx = 0
        dOH(t,0)/dx = 0
        OH(t,n) = OH_cathode_rate * t + OH_0
        φ(t,0) = φ_0
        φ(t,n) = 0

##   How to run
        julia poisson-nernst-planck-finite-difference.jl
        julia laplace-nernst-planck-finite-difference.jl
