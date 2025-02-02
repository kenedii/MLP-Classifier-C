#include "./include/feed_forward.h"

float squared_error(float y, float y_hat)
{
    return (y - y_hat) * (y - y_hat);
}

float mse(float *X, float *y, int n, float w00, float w01, float w10, float w11, float b00, float b01, float b1)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        sum += squared_error(y[i], feed_forward(X[i], w00, w01, w10, w11, b00, b01, b1));
    }
    return sum * (1.0 / 2 * n);
}
