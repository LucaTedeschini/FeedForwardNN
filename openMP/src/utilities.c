//
// Created by sbrodox on 15/05/25.
//
#include "network.h"
#include "utilities.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>


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
            network[i].nodes[j].delta = 0.0f;

            if (i < size - 1) {
                network[i].nodes[j].weights = malloc(sizeof(float) * layers_size[i + 1]);
                for (int z = 0; z < layers_size[i + 1]; z++) {
                    network[i].nodes[j].weights[z] = 0.1f;
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

float* get_output_layer(layer* network, int size) {
    float* results = malloc(sizeof(float) * network[size-1].size);
    for (int i=0; i < network[size-1].size; i++) {
        results[i] = network[size-1].nodes[i].value;
    }
    return softmax(results, network[size-1].size);
}

float activation_function(float value) {
    return value >= 0 ? value : 0;
}

float derivative_activation_function(float value) {
    return value > 0 ? 1.0f : 0.0f;
}
//unused, simple formula used
float cross_entropy_loss(const float* prediction, const float* real, int size) {
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        if (real[i] > 0.0f) {
            loss -= real[i] * logf(prediction[i] + 1e-9f);  // avoid log(0)
        }
    }
    return loss;
}

float* softmax(const float* values, int size) {
    float* result = malloc(sizeof(float) * size);
    if (values != NULL) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            // Avoids overflows

            result[i] = expf(values[i]);
            sum += result[i];
        }

        for (int i = 0; i < size; i++) {
            result[i] /= sum;
        }
    } //TODO: make else condition, but it should never land in else
    return result;
}

float* forward_pass(layer* network, int size){
    for (int i=0; i<size-1; i++) {
        // TODO: Better logic here, DRY and memory management
        float* accumulator = malloc(sizeof(float) * network[i+1].size);
        for (int j=0; j<network[i+1].size; j++) accumulator[j] = 0.0f;

        for (int j=0; j<network[i].size; j++) {
            for (int z = 0; z < network[i+1].size; z++) {
                accumulator[z] += network[i].nodes[j].weights[z] * network[i].nodes[j].value;
            }
        }
        for (int j=0; j<network[i+1].size; j++) {
            network[i+1].nodes[j].value = activation_function(accumulator[j] + network[i+1].nodes[j].bias);
        }

        free(accumulator);
    }

    return get_output_layer(network, size);
}

void backward_pass(layer* network, int size, const float* actual_value, const float* output_results) {
    for (int i=0; i < network[size-1].size; i++) {
        float y = actual_value[i];
        float o = output_results[i];

        //simplified formula, output_results comes from a softmax. Otherwise Jacobians would be involved.
        //TODO: use the complete formula
        network[size-1].nodes[i].delta = o - y;
    }
}

void backpropagation(layer* network, int size, float learning_rate) {
    for (int l = size - 2; l >= 0; l--) {
        for (int i = 0; i < network[l].size; i++) {
            float sum = 0.0f;
            for (int j = 0; j < network[l+1].size; j++) {
                // Sum the contributes
                sum += network[l].nodes[i].weights[j] * network[l+1].nodes[j].delta;
            }
            float val = network[l].nodes[i].value;
            //Compute the delta, multiplying contributes with the derivative
            network[l].nodes[i].delta = derivative_activation_function(val) * sum;
        }

        // Update weights and biases
        for (int i = 0; i < network[l].size; i++) {
            for (int j = 0; j < network[l+1].size; j++) {
                // Gradient descent update
                network[l].nodes[i].weights[j] -= learning_rate * network[l].nodes[i].value * network[l+1].nodes[j].delta;
            }
        }

        for (int i = 0; i < network[l+1].size; i++) {
            network[l+1].nodes[i].bias -= learning_rate * network[l+1].nodes[i].delta;
        }
    }
}



