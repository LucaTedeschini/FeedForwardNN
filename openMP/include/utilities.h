//
// Created by sbrodox on 15/05/25.
//


#ifndef OPENMP_UTILITIES_H
#define OPENMP_UTILITIES_H

#include <stdbool.h>
#include "network.h"

float random_float(void);
layer* create_network(const int* layers_size, int size);
void fill_input_layer(layer* network, const float* values);
float* get_output_layer(layer* network, int size);
float activation_function(float value);
float derivative_activation_function(float value);
float cross_entropy_loss(const float* prediction, const int* real, int size);
float* softmax(float* values, int size);
float* forward_pass(layer* network, int size);
void backward_pass(layer* network, int size, const int* actual_value, const float* output_results);
void backpropagation(layer* network, int size, float learning_rate);
void one_hot_encoding(int* encoded, int value, int size);
bool check_correctness(float* prediction, int* truth, int lenght);
float* matrix_mul(float** weights, const float* values, const float* biases,float* result, int input_size, int output_size);


#endif //OPENMP_UTILITIES_H
