/* Calculates the derivative of the weights and bias connected to the
output neuron (layer 1). (Updates weights w10, w11, and b1)
*/
float dl1bias(float y, float y_hat)
{
    return -(y - y_hat) * y_hat * (1 - y_hat);
}

float dl1w(float x, float y, float y_hat, float w, float b)
/* Float w1: weight connected to output neuron that proceeds
the weight we are updating */
{
    return -(y - y_hat) * y_hat * (1 - y_hat) * a_neuron(x, w, b);
}
/* Calculates the derivative of the weights and bias connected to the
hidden layer neurons (layer 0). (Updates weights w00, w01, b00, b01)
*/
float dl0bias(float y, float y_hat, float x, float w, float b, float w1)
{
    if (x * w + b > 0)
    {
        return -(y - y_hat) * y_hat * (1 - y_hat) * w1 * 1;
    }
    else
    {
        return 0;
    }
}

float dl0weight(float y, float y_hat, float x, float w, float b, float w1)
{
    if (x * w + b > 0)
    {
        return -(y - y_hat) * y_hat * (1 - y_hat) * w1 * x * 1;
    }
    else
    {
        return 0;
    }
}