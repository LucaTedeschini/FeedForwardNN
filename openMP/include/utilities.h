//
// Created by sbrodox on 15/05/25.
//


#ifndef OPENMP_UTILITIES_H
#define OPENMP_UTILITIES_H
#include "network.h"


layer* create_network(const int* layers_size, int size);
void fill_input_layer(layer* network);
void print_output_layer(layer* network, int size);


#endif //OPENMP_UTILITIES_H
