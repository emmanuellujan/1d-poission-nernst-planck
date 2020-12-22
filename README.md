## [WIP] 1D Poisson/Laplace-Nernst-Planck equations solution using NeuralPDE and the Finite Difference method


###   Mathematical model


####   Equations
        d2φ/dx2 = -F/epsilon * ( z_H * H + z_OH * OH)   or  d2φ/dx2 = 0 
        dNa/dt =  ( 1.0 / Pe_Na ) * d2Na/dx2 
                 + z_Na / ( abs(z_Na) * M_Na ) * ( dNa/dx * dφ/dx + Na * d2φ/dx2 )
        dCl/dt = ( 1.0 / Pe_Cl ) * d2Cl/dx2
                 + z_Cl / ( abs(z_Cl) * M_Cl ) * ( dCl/dx * dφ/dx + Cl * d2φ/dx2 )
        dH/dt =  ( 1.0 / Pe_H ) * d2H/dx2 
                 + z_H / ( abs(z_H) * M_H ) * ( dH/dx * dPhi/dx + H * d2φ/dx2 )
                 + k_wb * H2O - k_wf * H * OH
        dOH/dt =  ( 1.0 / Pe_OH ) * d2OH/dx2 
                 + z_H / ( abs(z_OH)*M_OH ) * ( dOH/dx * dφ/dx + OH * d2φ/dx2 )
                 + k_wb * H2O - k_wf * H * OH

#### Initial conditions:
        Na(0,x) = Na_0
        Cl(0,x) = Cl_0
        H(0,x) = H_0
        OH(0,x) = OH_0

#### Boundary conditions:

Butler-Volmer equations have been replaced by a linear approximation.

        Na(t,0) = 0.0
        Na(t,n) = 2.0 * Na_0
        Cl(t,0) = 1.37 * Cl_0
        Cl(t,n) = 0.0
        H(t,0) = 1.25 * H_0
        H(t,n) = H_0
        OH(t,0) = OH_0
        OH(t,n) = 1.25 * OH_0
        φ(t,0) = φ_0
        φ(t,x_max) = φ_1
        
#### References 

- "pH front tracking in the electrochemical treatment (EChT) of tumors: Experiments and simulations", 
P. Turjanski, N. Olaiz, P. Abou-Adal, C. Suárez 1 , M. Risk 1 , G. Marshall". Electrochimica Acta 54 (2009) 6199–6206.

- "Electroterapia y Electroporación en el tratamiento de tumores: modelos teóricos y experimentales". P. Turjanski. Departamento de Computación. Facultad de Ciencias Exactas y Naturales. Universidad de Buenos Aires. 2011.

        
### Installation and running in GNU/Linux

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
        $ cd 1d-poission-nernst-planck
        $ julia 1d-poisson-nernst-planck-adim-neuralpde.jl
        $ julia 1d-laplace-nernst-planck-adim-neuralpde.jl
        $ julia 1d-poisson-nernst-planck-adim-finite-difference.jl
        $ julia 1d-laplace-nernst-planck-adim-finite-difference.jl
    ```
