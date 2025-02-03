#include "./include/feed_forward.h"

/* Calculates the derivative of the weights and bias connected to the
   output neuron (layer 1). (Updates weights w10, w11, and b1) */
float dl1bias(float y, float y_hat)
{
    // BCE derivative for output layer bias
    return (y_hat - y);
}

float dl1w(float x, float y, float y_hat, float w, float b)
{
    // BCE derivative for output layer weight (w10 or w11)
    return (y_hat - y) * a_neuron(x, w, b);
}

/* Calculates the derivative of the weights and bias connected to the
   hidden layer neurons (layer 0). (Updates weights w00, w01, b00, b01) */
float dl0bias(float y, float y_hat, float x, float w, float b, float w1)
{
    // Derivative of the hidden layer bias
    float relu_grad = (x * w + b > 0) ? 1 : 0; // ReLU derivative

    return (y_hat - y) * w1 * relu_grad;
}

float dl0weight(float y, float y_hat, float x, float w, float b, float w1)
{
    // Derivative of the hidden layer weight
    float relu_grad = (x * w + b > 0) ? 1 : 0; // ReLU derivative

    return (y_hat - y) * w1 * relu_grad * x;
}
