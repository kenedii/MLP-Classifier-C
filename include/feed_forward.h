#ifndef FEED_FORWARD_H
#define FEED_FORWARD_H

// Function to compute the output of the neural network for a single input
float feed_forward(float x, float w00, float w01, float w10, float w11, float b00, float b01, float b1);

// Function to compute the output of a single neuron
float a_neuron(float x, float w, float b);

// Activation functions
float relu(float x);
float sigmoid(float x);

#endif // FEED_FORWARD_H