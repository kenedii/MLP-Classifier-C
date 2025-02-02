#include "./include/activations.h"

float feed_forward(float x, float w00, float w01, float w10, float w11, float b00, float b01, float b1)
{
    float neuron_00 = relu(x * w00 + b00);
    float neuron_01 = relu(x * w01 + b01);
    return sigmoid((neuron_00 * w10 + neuron_01 * w11 + b1));
}

float a_neuron(float x, float w, float b)
// Returns the output of a neuron
{
    return relu(x * w + b);
}