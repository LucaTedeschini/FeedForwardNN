//
// Created by sbrodox on 8/6/25.
//

#ifndef DATALOADER_H
#define DATALOADER_H

#define MAX_LAYERS 100

int read_config(const char* filename, int* train_size, int* test_size, int* epochs, int* layer_sizes, int* size);
#endif //DATALOADER_H
