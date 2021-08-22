# Normal form Autoencoder
 Constructing low-dimensional parameterized representations of high dimensional dynamics using normal forms as a building block by using an autoencoder framework. Neural network training is implemented with `Flux.jl`, a Julia library. Paper available [on arXiv](https://arxiv.org/abs/2106.05102)
 
 # Reproduce results
 Download datasets from [here](https://doi.org/10.4121/14790657.v1), and extract contents to `NormalFormAE/NFAEdata`. Use the scripts in `run` to reproduce the results from the paper.
 
 Note you need CUDA to run this package.
 
 ## How to use
 - If you have Linux/Mac, run the following on your terminal to install `Julia`  in one command 
 ```terminal
 bash -ci "$(curl -fsSL https://raw.githubusercontent.com/abelsiqueira/jill/master/jill.sh)"
 ```
 from [here](https://github.com/abelsiqueira/jill). 
 - Clone this package and enter the directory. Run `julia` on your terminal.
 - Now run the following:
```julia
 julia> ] activate .
 julia> ] instantiate
 ```
which will automatically install the necessary `Julia` packages you need.

- Run an example (tests coming soon) via the terminal or [REPL Shell mode](https://docs.julialang.org/en/v1/stdlib/REPL/#man-shell-mode). Note to run in the REPL Shell mode, you need to use the backspace/delete key to exit out of Pkg mode, and then type a `;`. Find out more about running Julia files [in the Julia docs](https://docs.julialang.org/en/v1/manual/getting-started/).
```terminal
julia -i run/run_nf.jl
```

