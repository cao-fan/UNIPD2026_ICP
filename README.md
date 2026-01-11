# To run the script a CUDA compatible GPU is required(see [CUDA.jl](https://cuda.juliagpu.org/stable/installation/overview/#InstallationOverview) for more info)

Open the terminal in the project folder
Install Julia 1.11.7(preferably through [JuliaUp](https://github.com/JuliaLang/juliaup))

> juliaup add 1.11.7
>
> julia +1.11.7

`cd(raw"<folder path>")` into the folder using the Julia REPL if you are not already inside the project folder.

Enter Package Mode by pressing ']' and then run:

> activate .
> 
> instantiate

To run the program run:

> include("script.jl")
