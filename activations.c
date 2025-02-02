// Contains activation functions and their derivatives

float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

float dsigmoid(float x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

float relu(float x)
{
    return x > 0 ? x : 0;
}

float drelu(float x)
{
    return x > 0 ? 1 : 0;
}