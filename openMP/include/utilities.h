//
// Created by sbrodox on 15/05/25.
//


#ifndef OPENMP_UTILITIES_H
#define OPENMP_UTILITIES_H
#include "network.h"


layer* create_network(const int* layers_size, int size);
void fill_input_layer(layer* network);
float* get_output_layer(layer* network, int size);
float activation_function(float value);
float derivative_activation_function(float value);
float cross_entropy_loss(const float* prediction, const float* real, int size);
float* softmax(const float* values, int size);
float* forward_pass(layer* network, int size);
void backward_pass(layer* network, int size, const float* actual_value, const float* output_results);
void backpropagation(layer* network, int size, float learning_rate);


#endif //OPENMP_UTILITIES_H
