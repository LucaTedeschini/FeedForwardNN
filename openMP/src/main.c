#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "network.h"
#include "utilities.h"
#include "dataloader.h"
#include "readconfig.h"




int main(void) {
    srand(time(NULL));

    int train_size, test_size, epochs;
    int layer_sizes[MAX_LAYERS];
    int size;
    int total_weights, total_nodes;
    printf("Reading configurations...\n");
    read_config("../config.txt", &train_size, &test_size, &epochs, layer_sizes, &size);
    printf("Done!\n");
    printf("Network has %i layers\n",size);
    printf("[ ");
    for (int i = 0; i<size; i++) {
        total_nodes+=layer_sizes[i];
        if (i < size-1) total_weights += layer_sizes[i] * layer_sizes[i+1];
        printf("%i ",layer_sizes[i]);
    }
    printf("] \n");

    printf("Network has %i total weights and %i total nodes\n",total_weights,total_nodes);


    layer* network = create_network(layer_sizes, size);

    float **X_train, **X_test;
    int *Y_train, *Y_test;
    read_dataset(&X_train, &Y_train, true);
    read_dataset(&X_test, &Y_test, false);

    float error = 0;
    int* true_encoded = malloc((sizeof(int) * 10));
    float* results;

    int *shuffled_indices = malloc(train_size * sizeof(int));
    for(int k=0; k<train_size; ++k) shuffled_indices[k] = k;

    // Epochs
    for (int i=0; i<epochs; i++) {
        printf("Running epoch %i / %i\n", i ,epochs);
        error = 0;
        for (int k = train_size - 1; k > 0; k--) {
            int rand_idx = rand() % (k + 1);
            int temp = shuffled_indices[k];
            shuffled_indices[k] = shuffled_indices[rand_idx];
            shuffled_indices[rand_idx] = temp;
        }
        printf("-----------\n");
        for (int j=0; j < train_size; j++) {
            if (j % 6000 == 0) {
                printf("+");
                fflush(stdout);
            }
            fill_input_layer(network, X_train[shuffled_indices[j]]);
            results = forward_pass(network, size);
            one_hot_encoding(true_encoded, Y_train[shuffled_indices[j]], 10);
            backward_pass(network, size, true_encoded, results);
            backpropagation(network, size, 0.01f);
            error += cross_entropy_loss(results, true_encoded, network[size-1].size);
        }

        printf("\n#####################\nEpoch %i - Loss %f\n", i, error / (float)train_size);
        int correct = 0;
        for (int j=0; j<test_size; j++) {
            fill_input_layer(network, X_test[j]);
            results = forward_pass(network, size);
            one_hot_encoding(true_encoded, Y_test[j], 10);

            if (check_correctness(results, true_encoded, 10))
                correct++;
        }
        printf("Accuracy : %f%%", ((double)correct / test_size) * 100);
        printf("\n#####################\n");

    }

    free(shuffled_indices);

    return 0;
}
