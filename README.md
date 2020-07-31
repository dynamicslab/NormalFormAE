# Normal form Autoencoder
 Constructing low-dimensional parameterized representations of high dimensional dynamics using normal forms as a building block by using an autoencoder framework. Neural network training is implemented with `Flux.jl`, a Julia library.
 My notes are available [here](https://www.overleaf.com/read/vwqwrnpjvrtn).
 
 Note you need CUDA to run this package.
 
 ## How to use
 - If you have Linux/Mac, run the following on your terminal to install `Julia`  in one command 
 ```
 bash -ci "$(curl -fsSL https://raw.githubusercontent.com/abelsiqueira/jill/master/jill.sh)"
 ```
 from [here](https://github.com/abelsiqueira/jill). 
 - Clone this package and enter the directory. Run `julia` on your terminal.
 - Now run the following:
```
 ] activate .
 ] instantiate
 ```
which will automatically install the necessary `Julia` packages you need.

- Run an example via   
```
julia -i run/run_Hopf.jl
```
## Note on branches
- `master` contains the standard AE + NLRAN + sensitivity analysis 
- branch `Kathleen` implements the [(Champion et al, 2019)](https://www.pnas.org/content/116/45/22445.abstract) model execpt for switching SINDy with an explicit RHS of the latent dynamics.
