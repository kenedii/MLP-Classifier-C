#ifndef BACKPROP_H
#define BACKPROP_H

/* Calculates the derivative of the bias connected to the
output neuron (layer 1). (Updates bias b1)
*/
float dl1bias(float y, float y_hat);

/* Calculates the derivative of the weights connected to the
output neuron (layer 1). (Updates weights w10, w11)
*/
float dl1w(float x, float y, float y_hat, float w, float b);

/* Calculates the derivative of the bias connected to the
hidden layer neurons (layer 0). (Updates biases b00, b01)
*/
float dl0bias(float y, float y_hat, float x, float w, float b, float w1);

/* Calculates the derivative of the weights connected to the
hidden layer neurons (layer 0). (Updates weights w00, w01)
*/
float dl0weight(float y, float y_hat, float x, float w, float b, float w1);

#endif // BACKPROP_H