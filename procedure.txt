Well, how does a Neural network learn?
1. It all starts with a vector of inputs.
2. Then this input vector gets multiplied with a dimensionally equal vector of weights. Result = a number for each neuron
3. We then sum the output of each neuron and give it to the next layer and redo the process as the output of the current layer becomes the input of the next layer. (Now we, again, have two vectors of values to apply matrix multiplication on.)
4. We keep doing this until we get the output number or vector depending on the problem.

"Now the Question is, how does the set of inputs translate into the output value? how does the relation work by multiplying and summing the outputs of these weights and neurons generally?"

Current Answer: The weights should be modified by each iteration in order for the input to result in the output eventually.

8(x) * 0.5(w) > 2 => Decrease w by dw/dE
Now:
8(x) * 0.25(w) = 2 => Modify w by 0

We need some pen-and-paper work.

Why don't we just simplify the process to the minimum. One layer of 2 neurons, Input vector size of 2, another layer of output size 1x1. Backpropagate with the activation function of sigmoid. Output activation linear.

