#include <stdio.h>
#include "./include/feed_forward.h"
#include "./include/mse.h"
#include "./include/backprop.h"

// Housing prices dataset
float X[] = {8450, 9600, 11250, 9550, 14260, 14115, 10084, 10382, 6120, 7420};
float Y[] = {208500, 181500, 223500, 140000, 250000, 143000, 307000, 200000, 129900, 118000};
float lr = 0.01;
float num_epochs = 100;
int n = sizeof(X) / sizeof(X[0]);

typedef struct
{
    float b00;
    float b01;
    float b1;
    float w00;
    float w01;
    float w10;
    float w11;
    float mse;
} current_weights;

typedef struct
{
    float best_b00;
    float best_b01;
    float best_b1;
    float best_w00;
    float best_w01;
    float best_w10;
    float best_w11;
    float best_mse;
} best_weights;

void train_model(float *X, float *Y, float lr, int num_epochs, int n)
{
    current_weights w = {.b00 = 0, .b01 = 0, .b1 = 0, .w00 = 0, .w01 = 0, .w10 = 0, .w11 = 0, .mse = 0};
    best_weights b = {.best_b00 = 0, .best_b01 = 0, .best_b1 = 0, .best_w00 = 0, .best_w01 = 0, .best_w10 = 0, .best_w11 = 0, .best_mse = 1000000};

    printf("Beginning training . . .\n");

    for (int i = 0; i <= num_epochs; i++)
    {

        for (int j = 0; i <= n; j++)
        {
            float x1 = X[j];
            float y = Y[j];

            float y_hat = feed_forward(x1, w.w00, w.w01, w.w10, w.w11, w.b00, w.b01, w.b1);

            w.b1 = w.b1 - lr * dl1bias(y, y_hat);
            w.w10 = w.w10 - lr * dl1w(x1, y, y_hat, w.w00, w.b00);
            w.w11 = w.w11 - lr * dl1w(x1, y, y_hat, w.w01, w.b01);
            w.b00 = w.b00 - lr * dl0bias(y, y_hat, x1, w.w00, w.b00, w.w10);
            w.b01 = w.b01 - lr * dl0bias(y, y_hat, x1, w.w01, w.b01, w.w11);
            w.w00 = w.w00 - lr * dl0weight(y, y_hat, x1, w.w00, w.b00, w.w10);
            w.w01 = w.w01 - lr * dl0weight(y, y_hat, x1, w.w01, w.b01, w.w11);
        }
        w.mse = mse(X, Y, n, w.w00, w.w01, w.w10, w.w11, w.b00, w.b01, w.b1);
        printf("Epoch: %d, MSE: %f\n, Best MSE: %f", i, w.mse, b.best_mse);
        printf("Current weights: b00: %f, b01: %f, w00: %f, b1: %f, w10: %f, w11: %f\n", w.b00, w.b01, w.w00, w.b1, w.w10, w.w11);

        if (w.mse < b.best_mse) // Update best weights
        {
            b.best_mse = w.mse;
            b.best_b00 = w.b00;
            b.best_b01 = w.b01;
            b.best_w00 = w.w00;
            b.best_w01 = w.w01;
            b.best_w10 = w.w10;
            b.best_w10 = w.w10;
            b.best_w11 = w.w10;
        }
    }
    printf("Training complete . . .\n");
}

float main()
{
    train_model(X, Y, lr, num_epochs, n);

    // Wait for any key press
    printf("Press Enter to continue...");
    getchar(); // Wait for Enter key

    return 0;
}