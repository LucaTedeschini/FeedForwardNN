#include <stdio.h>
#include <stdlib.h>

#include "network.h"
#include "utilities.h"

//TODO: load this from a config file maybe?
const int layer_sizes[] = {3,5};
const int size = 2;


int main(void) {
    layer* network = create_network(layer_sizes, size);

    fill_input_layer(network);

    // Forward pass
    for (int i=0; i<size-1; i++) {
        // TODO: Better logic here, DRY and memory management
        float* accumulator = malloc(sizeof(float) * network[i+1].size);
        for (int j=0; j<network[i+1].size; j++) accumulator[j] = 0.0f;

        for (int j=0; j<network[i].size; j++) {
            for (int z = 0; z < network[i+1].size; z++) {
                accumulator[z] += network[i].nodes[j].weights[z] * network[i].nodes[j].value;
            }
        }

        //TODO: add activation function
        for (int j=0; j<network[i+1].size; j++) {
            network[i+1].nodes[j].value = accumulator[j] + network[i+1].nodes[j].bias;
        }
        free(accumulator);
    }

    print_output_layer(network, size);
    printf("Hello, World!\n");
    return 0;
}
