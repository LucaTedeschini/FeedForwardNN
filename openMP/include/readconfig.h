//
// Created by sbrodox on 16/05/25.
//

#ifndef OPENMP_READCONFIG_H
#define OPENMP_READCONFIG_H
#define MAX_LAYERS 100

int read_config(const char* filename, int* train_size, int* test_size, int* epochs, int* layer_sizes, int* size);

#endif //OPENMP_READCONFIG_H
