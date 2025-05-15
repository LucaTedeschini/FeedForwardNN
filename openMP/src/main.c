#include <stdio.h>
#include <stdlib.h>

#include "network.h"
#include "utilities.h"

//TODO: load this from a config file maybe?
const int layer_sizes[] = {3, 5, 10, 5, 3};
const int size = 5;


int main(void) {
    layer* network = create_network(layer_sizes, size);

    fill_input_layer(network);
    float actual_value[] = {1,0,0};
    // Forward pass
    for (int i=0; i<10; i++) {
        float* results = forward_pass(network, size);
        backward_pass(network, size, actual_value, results);
        backpropagation(network, size, 0.1f);

        float error = cross_entropy_loss(results, actual_value, network[size-1].size);
        printf("Epoch %i - Loss %f\n", i, error);
        for (int j=0; j<3; j++) {
            printf("\tNode %i output value: %f\n", j, results[j]);
        }
    }

    return 0;
}
