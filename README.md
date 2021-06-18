# Normal form Autoencoder
 Constructing low-dimensional parameterized representations of high dimensional dynamics using normal forms as a building block by using an autoencoder framework. Neural network training is implemented with `Flux.jl`, a Julia library. Paper available [on arXiv](https://arxiv.org/abs/2106.05102)
 
 # Reproduce results
 Download datasets from [here](https://doi.org/10.4121/14790657.v1), and extract contents to `NormalFormAE/NFAEdata`. Use the scripts in `run` to reproduce the results from the paper.
 
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

- Run an example (tests coming soon) via   
```
julia -i run/run_nf.jl
```

