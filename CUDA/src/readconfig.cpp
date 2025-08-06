//
// Created by sbrodox on 16/05/25.
//
#include <stdio.h>
#include <stdlib.h>
#include "readconfig.hpp"


int read_config(const char* filename, int* train_size, int* test_size, int* epochs, int* layer_sizes, int* size) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("ERROR OPENING FILE");
        exit(1);
    }

    //Parsing is not handled correctly, but I assume that the config file is syntatically correct
    if (fscanf(file, "%d", train_size) != 1 ||
        fscanf(file, "%d", test_size) != 1 ||
        fscanf(file, "%d", epochs) != 1) {
        fprintf(stderr, "ERROR READING TRAIN_SIZE / TEST_SIZE / EPOCHS \n");
        fclose(file);
        exit(1);
        }

    if (*train_size > 60000 || *train_size <= 0) {
        printf("TRAIN SIZE CANNOT EXCEED 60000 AND MUST BE GREATER THAN 0\n");
        exit(1);
    }

    if (*test_size > 10000 || *test_size <= 0) {
        printf("TEST SIZE CANNOT EXCEED 10000 AND MUST BE GREATER THAN 0\n");
        exit(1);
    }

    if (*epochs > 20 || *epochs <= 0) {
        printf("EPOCHS CANNOT EXCEED 20 AND MUST BE GREATER THAN 0\n");
        exit(1);
    }

    for (int i = 0; i < MAX_LAYERS; ++i) {
        if (fscanf(file, "%d", &layer_sizes[i]) != 1) {
            *size = i;
            break;
        }
    }

    if (layer_sizes[*size - 1] != 10) {
        printf("Last layer must have 10 nodes!\n");
        exit(1);
    }

    if (layer_sizes[0] != 784) {
        printf("First layer must have 784 nodes!\n");
        exit(1);
    }


    fclose(file);
    return 0;
}