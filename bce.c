#include "./include/feed_forward.h"
#include <math.h> // For log() function

// Binary cross-entropy cost for a single example
float bce_cost(float y, float y_hat)
{
    // To avoid log(0), we clamp the predictions between a small epsilon value and (1 - epsilon)
    const float epsilon = 0.000001;
    y_hat = fmaxf(epsilon, fminf(y_hat, 1.0 - epsilon)); // Clamp y_hat

    return -(y * log(y_hat) + (1 - y) * log(1 - y_hat));
}

// Calculate the average BCE cost for the entire dataset
float bce(float *X, float *y, int n, float w00, float w01, float w10, float w11, float b00, float b01, float b1)
{
    float sum = 0;
    for (int i = 0; i < n; i++)
    {
        float y_hat = feed_forward(X[i], w00, w01, w10, w11, b00, b01, b1);
        sum += bce_cost(y[i], y_hat);
    }
    return sum / n; // Return average BCE over all examples
}
