# MLP-Classifier-2-Layer
A MLP Classifier with functions for mse, binary cross entropy, relu, sigmoid, and weight updating through backprop.

**Model Description**:

**Input Layer:**

1 input node that receives the input data for 1 independent X feature variable.

**Hidden Layer:**

2 nodes in the hidden layer.

Each hidden node has its own weight and bias:

Node 1: Weight w00, Bias b00

Node 2: Weight w01, Bias b01

The output of each hidden node is calculated using the ReLU activation function (relu(x * w + b)), which is applied to the weighted sum of the input and the node's bias.

**Output Layer:**

The outputs of the two hidden nodes (neuron_00 and neuron_01) are combined to form the final input to the output layer:

The combined value is computed as neuron_00 * w10 + neuron_01 * w11 + b1, where w10 and w11 are weights, and b1 is a bias.

This combined value is then passed through the sigmoid activation function to produce the output (prediction y_hat).

The sigmoid output represents a probability (value between 0 and 1).
