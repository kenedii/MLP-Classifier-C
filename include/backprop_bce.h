#ifndef BACKPROP_BCE_H
#define BACKPROP_BCE_H

// Function to calculate the derivative of the bias connected to the output neuron (layer 1)
float dl1bias(float y, float y_hat);

// Function to calculate the derivative of the weights connected to the output neuron (layer 1)
float dl1w(float x, float y, float y_hat, float w, float b);

// Function to calculate the derivative of the bias connected to the hidden layer neurons (layer 0)
float dl0bias(float y, float y_hat, float x, float w, float b, float w1);

// Function to calculate the derivative of the weights connected to the hidden layer neurons (layer 0)
float dl0weight(float y, float y_hat, float x, float w, float b, float w1);

#endif // BACKPROP_BCE_H
