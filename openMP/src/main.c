#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "network.h"
#include "utilities.h"
#include "dataloader.h"

#define EPOCHS 10
#define TRAIN_SIZE 60000
#define TEST_SIZE 10000
//TODO: load this from a config file maybe?
const int layer_sizes[] = {IMAGE_SIZE, 500, 150, 10};
const int size = 4;


int main(void) {
    srand(time(NULL));

    layer* network = create_network(layer_sizes, size);

    float **X_train, **X_test;
    int *Y_train, *Y_test;
    read_dataset(&X_train, &Y_train, true);
    read_dataset(&X_test, &Y_test, false);

    float error = 0;
    int* true_encoded = malloc((sizeof(int) * 10));
    float* results;

    int *shuffled_indices = malloc(TRAIN_SIZE * sizeof(int));
    for(int k=0; k<TRAIN_SIZE; ++k) shuffled_indices[k] = k;

    // Epochs
    for (int i=0; i<EPOCHS; i++) {
        error = 0;
        for (int k = TRAIN_SIZE - 1; k > 0; k--) {
            int rand_idx = rand() % (k + 1);
            int temp = shuffled_indices[k];
            shuffled_indices[k] = shuffled_indices[rand_idx];
            shuffled_indices[rand_idx] = temp;
        }

        for (int j=0; j < TRAIN_SIZE; j++) {
            if (j % 5000 == 0) {
                printf("Processing %i / %i\n", j, TRAIN_SIZE);
            }
            fill_input_layer(network, X_train[shuffled_indices[j]]);
            results = forward_pass(network, size);
            one_hot_encoding(true_encoded, Y_train[shuffled_indices[j]], 10);
            backward_pass(network, size, true_encoded, results);
            backpropagation(network, size, 0.01f);
            error += cross_entropy_loss(results, true_encoded, network[size-1].size);
        }

        printf("\n#####################\nEpoch %i - Loss %f\n", i, error / TRAIN_SIZE);
        int correct = 0;
        for (int j=0; j<TEST_SIZE; j++) {
            fill_input_layer(network, X_test[j]);
            results = forward_pass(network, size);
            one_hot_encoding(true_encoded, Y_test[j], 10);

            if (check_correctness(results, true_encoded, 10))
                correct++;
        }
        printf("Accuracy : %f%%", ((double)correct / TEST_SIZE) * 100);
        printf("\n#####################\n");

    }

    free(shuffled_indices);

    return 0;
}
