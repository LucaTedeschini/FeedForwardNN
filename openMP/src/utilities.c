//
// Created by sbrodox on 15/05/25.
//
#include "network.h"
#include "utilities.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>


float random_float(void) {
    return ((float)rand() / (float)RAND_MAX) * 0.02f - 0.01f;
}

layer* create_network(const int* layers_size, int size) {
    layer* network = malloc(sizeof(layer) * size);

    // Instantiate each layer
    for (int i=0; i<size; i++) {
        network[i].nodes = malloc(sizeof(node) * layers_size[i]);
        network[i].size = layers_size[i];
        // Malloc the weight array and bias only for the first N-1 nodes


        // fill the nodes weights and bias
        for (int j = 0; j < layers_size[i]; j++) {
            network[i].nodes[j].bias = 0.1f;
            network[i].nodes[j].delta = 0.0f;

            if (i < size - 1) {
                network[i].nodes[j].weights = malloc(sizeof(float) * layers_size[i + 1]);
                for (int z = 0; z < layers_size[i + 1]; z++) {
                    network[i].nodes[j].weights[z] = random_float();
                }
            }
        }
    }

    return network;
}

void fill_input_layer(layer* network, const float* values){
    for (int i=0; i<network[0].size; i++) {
        network[0].nodes[i].value = values[i];
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
float cross_entropy_loss(const float* prediction, const int* real, int size) {
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
    float max_val = values[0];
    for (int i = 1; i < size; i++) {
        if (values[i] > max_val) max_val = values[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        result[i] = expf(values[i] - max_val); // *sottrai max_val*
        sum += result[i];
    }
    for (int i = 0; i < size; i++) {
        result[i] /= sum;
    }
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
            if (i == size - 2) {
                network[i+1].nodes[j].value = accumulator[j] + network[i+1].nodes[j].bias; // output layer: no activation
            } else {
                network[i+1].nodes[j].value = activation_function(accumulator[j] + network[i+1].nodes[j].bias); // hidden: ReLU
            }
        }

        free(accumulator);
    }

    return get_output_layer(network, size);
}

void backward_pass(layer* network, int size, const int* actual_value, const float* output_results) {
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

void one_hot_encoding(int* encoded, int value, int size) {
    for (int i=0; i<size; i++) {
        encoded[i] = 0;
    }

    encoded[value] = 1;

}

bool check_correctness(float* prediction, int* truth, int lenght) {
    int max_pred = 0;
    int max_truth = 0;
    float max_val = -1.0f;
    for (int i=0; i<lenght; i++) {
        if (prediction[i] > max_val) {
            max_pred = i;
            max_val = prediction[i];
        }
    }

    for (int i=0; i<lenght; i++) {
        if (truth[i] == 1) {
            max_truth = i;
        }
    }

    return max_truth == max_pred;

}


