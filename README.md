# Normal form Autoencoder
 Constructing low-rank parameterized representations of high dimensional dynamics using normal forms as a building block by using an autoencoder framework. Neural network training is implemented with `Flux.jl`, a Julia library.
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

- The file `Run` contains the pseudocode script to start training. To start training, run
```
chmod +x ./Run
./Run
```
## Note on branches
- branch `master` implements the new architecture (identfying a parameterized system).
- branch `Kathleen` implements the [(Champion et al, 2019)](https://www.pnas.org/content/116/45/22445.abstract) model by swicthing SINDy for an explicit RHS.

Feel free to change the `Run` file to your benefit and let me know of anything interesting! :)
