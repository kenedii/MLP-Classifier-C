#ifndef MSE_H
#define MSE_H

// Calculates the squared error for a single prediction vs actual value
float squared_error(float y, float y_hat);

// Calculates the Mean Squared Error for a set of predictions vs actual values
float mse(float *X, float *y, int n, float w00, float w01, float w10, float w11, float b00, float b01, float b1);

#endif // MSE_H