#include <stdio.h>

// Housing prices dataset
float X[] = {8450, 9600, 11250, 9550, 14260, 14115, 10084, 10382, 6120, 7420};
float Y[] = {208500, 181500, 223500, 140000, 250000, 143000, 307000, 200000, 129900, 118000};
float lr = 0.01;
float num_epochs = 100;
int n = sizeof(X) / sizeof(X[0]);

typedef struct
{
    float b0;
    float w01;
    float w02;
    float w10;
    float mse;
} current_weights;

typedef struct
{
    float best_b0;
    float best_w01;
    float best_w02;
    float best_w10;
    float best_mse;
} best_weights;

void train_model(float *X, float *Y, float lr, int num_epochs, int n)
{
    current_weights w = {.b0 = 0, .w01 = 0, .w02 = 0, .w10 = 0, .mse = 0};
    best_weights b = {.best_mse = 1000000000, .best_b0 = 0, .best_w01 = 0, .best_w02 = 0, .best_w10 = 0};

    printf("Beginning training . . .\n");

    for (int i = 0; i <= num_epochs; i++)
    {
        current_weights w = {.b0 = 1, .w01 = 1, .w02 = 1, .w10 = 1, .mse = 1};

        for (int j = 0; i <= n; j++)
        {
            float x1 = X[j];
            float x2 = X[j];
            float y = Y[j];

            float y_hat = feed_forward(x1, x2, w.w01, w.w02, w.w10, w.b0);
            float loss = squared_error(y, y_hat);

            float w.b0 = w.b0 - lr * dsquared_error(1, y, y_hat, 0);
            float dw_w01 = w.w01 - lr * dsquared_error(x1, y, y_hat, 1);
            float dw_w02 = w.w02 - lr * dsquared_error(x2, y, y_hat, 1);
            float dw_w10 = w.w10 - lr * dsquared_error(x1, y, y_hat, 1);
        }
        printf("Epoch: %d, MSE: %f\n, Best MSE: %f", i, mse(Y, y_hat, n));
        print("Current weights: b0: %f, w01: %f, w02: %f, w10: %f\n", w.b0, w.w01, w.w02, w.w10);

        if (mse(Y, y_hat, n) < b.best_mse)
        {
            b.best_mse = mse(Y, y_hat, n);
            b.best_b0 = w.b0;
            b.best_w01 = w.w01;
            b.best_w02 = w.w02;
            b.best_w10 = w.w10;
        }
    }
    printf("Training complete . . .\n");
}

float main()
{
    train_model(X, Y, lr, num_epochs, n);
    return 0;
}