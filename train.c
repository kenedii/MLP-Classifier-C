#include <stdio.h>
#include "./include/feed_forward.h"
#include "./include/bce.h"
#include "./include/backprop_bce.h"
#include <stdlib.h> // For rand()

float X[] = {8450, 9600, 11250, 9550, 14260, 14115, 10000, 10382, 6120, 7420, 15000, 8450, 9600, 11250, 9550, 14260, 14115, 10000, 10382, 6120, 7420};

// Binary target based on a threshold (200000 in this case)
float Y[] = {0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0}; // 0 = below 10000, 1 = above or equal 10000

float lr = 0.00001;
float num_epochs = 200;
int n = sizeof(X) / sizeof(X[0]);

// Function to generate a small random value between -0.01 and 0.01
float random_weight()
{
    return ((rand() / (float)RAND_MAX) * 0.2) - 0.1; // Between -0.1 and 0.1
}

typedef struct
{
    float b00;
    float b01;
    float b1;
    float w00;
    float w01;
    float w10;
    float w11;
    float loss;
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
    float best_loss;
} best_weights;

void train_model(float *X, float *Y, float lr, int num_epochs, int n)
{
    current_weights w = {
        .b00 = random_weight(),
        .b01 = random_weight(),
        .b1 = random_weight(),
        .w00 = random_weight(),
        .w01 = random_weight(),
        .w10 = random_weight(),
        .w11 = random_weight(),
        .loss = 1}; // Initialize random weights
    best_weights b = {.best_b00 = 0, .best_b01 = 0, .best_b1 = 0, .best_w00 = 0, .best_w01 = 0, .best_w10 = 0, .best_w11 = 0, .best_loss = 1000000};

    printf("Beginning training . . .\n");

    for (int i = 0; i < num_epochs; i++)
    {

        for (int j = 0; j < n; j++)
        {
            float x1 = X[j];
            float y = Y[j];

            float y_hat = feed_forward(x1, w.w00, w.w01, w.w10, w.w11, w.b00, w.b01, w.b1);
            printf("Prediction: %f, Actual: %f\n", y_hat, y);
            w.b1 = w.b1 - lr * dl1bias(y, y_hat);
            w.w10 = w.w10 - lr * dl1w(x1, y, y_hat, w.w00, w.b00);
            printf("updating weight by dl1w(x1, y, y_hat, w.w00, w.b00); %f\n", dl1w(x1, y, y_hat, w.w00, w.b00));
            w.w11 = w.w11 - lr * dl1w(x1, y, y_hat, w.w01, w.b01);
            w.b00 = w.b00 - lr * dl0bias(y, y_hat, x1, w.w00, w.b00, w.w10);
            w.b01 = w.b01 - lr * dl0bias(y, y_hat, x1, w.w01, w.b01, w.w11);
            w.w00 = w.w00 - lr * dl0weight(y, y_hat, x1, w.w00, w.b00, w.w10);
            w.w01 = w.w01 - lr * dl0weight(y, y_hat, x1, w.w01, w.b01, w.w11);
        }
        w.loss = bce(X, Y, n, w.w00, w.w01, w.w10, w.w11, w.b00, w.b01, w.b1);
        printf("Epoch: %d, BCE: %f, Best BCE: %f\n", i, w.loss, b.best_loss);
        printf("Current weights: b00: %f, b01: %f, b1: %f, w00: %f, w01: %f, w10: %f, w11: %f\n", w.b00, w.b01, w.b1, w.w00, w.w01, w.w10, w.w11);

        if (w.loss < b.best_loss) // Update best weights
        {
            b.best_loss = w.loss;
            b.best_b00 = w.b00;
            b.best_b01 = w.b01;
            b.best_w00 = w.w00;
            b.best_w01 = w.w01;
            b.best_w10 = w.w10;
            b.best_w11 = w.w11;
            b.best_b1 = w.b1;
        }
    }
    printf("Training complete . . .\n");
    printf("Best BCE: %f\n", b.best_loss);
    printf("Best weights: b00: %f, b01: %f, b1: %f, w00: %f, w01: %f, w10: %f, w11: %f\n", b.best_b00, b.best_b01, b.best_b1, b.best_w00, b.best_w01, b.best_w10, b.best_w11);
}

float main()
{
    train_model(X, Y, lr, num_epochs, n);

    // Wait for any key press
    printf("Press Enter to continue...");
    getchar(); // Wait for Enter key

    return 0;
}