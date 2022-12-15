## Model 1

Model 1 is the simplist test of the idea, given a grid we cells, we define the neural network similar to a cellular automata, without backwards connections and without sparsification techniques.

Model 2 is the prelimary test of this whole idea. Model 2 just uses regular backprop on a geometric grid of cells which forms a sparse directed acrylic graph implicitly through the way that its designed. The backprop is very vanilla, a straight forward pass and backward pass, consecutive in nature. 