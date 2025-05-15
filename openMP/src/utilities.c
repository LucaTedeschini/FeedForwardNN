//
// Created by sbrodox on 15/05/25.
//
#include "network.h"
#include "utilities.h"
#include <stdlib.h>


layer* create_network(const int* layers_size, int size) {
    layer* network = malloc(sizeof(layer) * size);

    // Instantiate each layer
    for (int i=0; i<size; i++) {
        network[i].nodes = malloc(sizeof(node) * layers_size[i]);
        network[i].size = layers_size[i];
        // Malloc the weight array and bias only for the first N-1 nodes
        if (i < size-1) {

            // fill the nodes weights and bias
            for (int j = 0; j < layers_size[i]; j++) {
                network[i].nodes[j].bias = 0.2f;
                network[i].nodes[j].weights = malloc(sizeof(float) * layers_size[i + 1]);
                for (int z = 0; z < layers_size[i + 1]; z++) {
                    network[i].nodes[j].weights[z] = z;
                }
            }
        }

    }

    return network;
}
