## [WIP] These codes solve 1D Poisson-Nernst-Planck and Laplace-Nernst-Planck equations using NeuralPDE and the Finite Difference method

The equation system is based on "pH front tracking in the electrochemical treatment (EChT) of tumors: Experiments and simulations", 
P. Turjanski, N. Olaiz, P. Abou-Adal, C. Suárez 1 , M. Risk 1 , G. Marshall". Electrochimica Acta 54 (2009) 6199–6206.

1D Poisson-Nernst-Planck and Laplace-Nernst-Planck

###   Model

####   Equations
        d2φ/dx2 = -F/epsilon * ( z_H * H + z_OH * OH)   or  d2φ/dx2 = 0 
        dH/dt =  D_H * d2H/dx2 
                 + c_H * ( dH/dx * dφ/dx + H * d2φ/dx2 )
                 + k_wb * H2O - k_wf * H * OH
        dOH/dt = D_OH * d2OH/dx2
                 + c_OH * ( dOH/dx * dφ/dx + OH * d2φ/dx2 )
                 + k_wb * H2O - k_wf * H * OH

####   Initial conditions
        H(0,x) = H_0
        OH(0,x) = OH_0
        φ(0,x) = 0

####   Boundary conditions

Butler-Volmer equations have been replaced by a linear approximation.

        H(t,0) = H_anode_rate * t + H_0
        dH(t,n)/dx = 0
        dOH(t,0)/dx = 0
        OH(t,n) = OH_cathode_rate * t + OH_0
        φ(t,0) = φ_0
        φ(t,n) = 0
        
### Installation and running

1) Download Julia from https://julialang.org/downloads/

    E.g.
    ```
        $ wget https://julialang-s3.julialang.org/bin/linux/x64/1.5/julia-1.5.3-linux-x86_64.tar.gz
     ```
2) Extract file
     ```
        $ tar xvzf  julia-1.5.3-linux-x86_64.tar.gz
     ```
3) Copy to /opt and create link
     ```
        $ sudo mv  ./julia-1.5.3 /opt/
        $ sudo ln -s /opt/julia-1.5.3/bin/julia /usr/local/bin/julia
     ```
4) Install required packets
    ```
        $ julia
        julia> import Pkg
        julia> Pkg.add("NeuralPDE")
        julia> Pkg.add("Flux")
        julia> Pkg.add("ModelingToolkit")
        julia> Pkg.add("GalacticOptim")
        julia> Pkg.add("Optim")
        julia> Pkg.add("DiffEqFlux")
        julia> Pkg.add("Plots")
        julia> Pkg.add("Quadrature")
        julia> Pkg.add("Cubature")
        julia> Pkg.add("Cuba")
        julia> Pkg.add("LaTeXStrings")
    ```
     
4) Clone project directory
    ```
        $ git clone https://github.com/emmanuellujan/1d-poission-nernst-planck.git
     ```

4) Run
    ```
        $ julia poisson-laplace-nernst-planck-neuralpde.jl
        $ julia laplace-nernst-planck-neuralpde.jl
        $ julia poisson-nernst-planck-finite-difference.jl
        $ julia laplace-nernst-planck-finite-difference.jl
    ```
