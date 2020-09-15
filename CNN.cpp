#include "layers.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

int main(int argc, char **argv) {
    FILE *fpt, *fpt2, *fout;
    char header[80];
    int rows, cols, max;
    int **images, *labels;
    unsigned char throwaway;
    int r, c, i, j, f;

    // hyperparameters
    int num_images = 1200;
    int num_train = 1000;
    float learning_rate = 0.005;
    int per_print = 100;
    int num_epochs = 1;
    int num_filters = 8;
    int filter_size = 3;

    /************************************************ allocate memory ************************************************/

    // allocate image and label arrays
    images = (int **) calloc(num_images, sizeof(int *));
    for (i = 0; i < num_images; i++) {
        images[i] = (int *) calloc(28*28, sizeof(int));
    }
    labels = (int *) calloc(num_images, sizeof(int));

    // read in images and labels
    fpt = fopen("mnist_images_full.txt", "rb");
    fpt2 = fopen("mnist_labels_full.txt", "rb");
    for (i = 0; i < num_images; i++) {
        fscanf(fpt2, "%d\n", &labels[i]);
        for (j = 0; j < 28*28; j++) {
            fscanf(fpt, "%d,", &images[i][j]);
        }
        fscanf(fpt, "%c", &throwaway);
    }

    fclose(fpt);
    fclose(fpt2);

    // read in initial filter weights
    float *filters_init;
    filters_init = (float *) calloc(8 * 9, sizeof(float));
    fpt = fopen("filters_init.txt", "rb");
    for (i = 0; i < 8 * 9; i++) {
        fscanf(fpt, "%f\n", &filters_init[i]);
    }
    fclose(fpt);

    // read in initial softmax weights
    float *soft_weight_init;
    soft_weight_init = (float *) calloc(13*13*8*10, sizeof(float));
    fpt = fopen("soft_weights.txt", "rb");
    for (i = 0; i < 13*13*8*10; i++) {
        fscanf(fpt, "%f\n", &soft_weight_init[i]);
    }
    fclose(fpt);

    // set initial softmax biases
    float *soft_bias_init;
    soft_bias_init = (float *) calloc(10, sizeof(float));
    for (i = 0; i < 10; i++) {
        soft_bias_init[i] = 0.0;
    }

    /************************************************ initialize layers ************************************************/

    Conv_layer conv(28, 28, num_filters, filter_size, learning_rate);
    Maxpool_layer maxpool(26, 26, num_filters);
    Softmax_layer softmax(13*13*num_filters, 10, learning_rate);


    return 0;
}