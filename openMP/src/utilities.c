//
// Created by sbrodox on 15/05/25.
//
#include "network.h"
#include "utilities.h"
#include <stdlib.h>
#include <stdio.h>


layer* create_network(const int* layers_size, int size) {
    layer* network = malloc(sizeof(layer) * size);

    // Instantiate each layer
    for (int i=0; i<size; i++) {
        network[i].nodes = malloc(sizeof(node) * layers_size[i]);
        network[i].size = layers_size[i];
        // Malloc the weight array and bias only for the first N-1 nodes

        // fill the nodes weights and bias
        for (int j = 0; j < layers_size[i]; j++) {
            network[i].nodes[j].bias = 0.2f;
            if (i < size - 1) {
                network[i].nodes[j].weights = malloc(sizeof(float) * layers_size[i + 1]);
                for (int z = 0; z < layers_size[i + 1]; z++) {
                    network[i].nodes[j].weights[z] = z;
                }
            }
        }
    }

    return network;
}

void fill_input_layer(layer* network){
    for (int i=0; i<network[0].size; i++) {
        network[0].nodes[i].value = i;
    }
}

void print_output_layer(layer* network, int size) {
    for (int i=0; i < network[size-1].size; i++) {
        printf("Node %i value: %f\n", i, network[size-1].nodes[i].value);
    }
}

