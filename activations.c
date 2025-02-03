#include <math.h>
// Contains activation functions

float sigmoid(float x)
{
    return 1.0f / (1.0f + exp(-x));
}

float relu(float x)
{
    return x > 0 ? x : 0;
}
