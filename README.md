# Normal form Autoencoder
 Constructing low-dimensional parameterized representations of high dimensional dynamics using normal forms as a building block by using an autoencoder framework. Neural network training is implemented with `Flux.jl`, a Julia library. WIP paper available [here](https://inductive-biases.github.io/papers/46.pdf)
 
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

