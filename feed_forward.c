float feed_forward(float x, float w00, float w01, float w10, float b0, float b1)
{
    float first_layer = (x * w00) + (x * w01) + b0;
    float second_layer = first_layer * w10 + b1;
    return second_layer;
}