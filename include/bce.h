#ifndef BCE_H
#define BCE_H

// Function to compute the binary cross-entropy cost for a single example
float bce_cost(float y, float y_hat);

// Function to compute the average binary cross-entropy cost for the entire dataset
float bce(float *X, float *y, int n, float w00, float w01, float w10, float w11, float b00, float b01, float b1);

#endif // BCE_H
