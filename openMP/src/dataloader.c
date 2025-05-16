//
// Created by sbrodox on 16/05/25.
//
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include "dataloader.h"


void read_dataset(float*** X, int** Y, bool is_train) {
    int length = is_train ? 60000 : 10000;

    char fpath[30];
    if (is_train){
        strcpy(fpath,"../../MNIST/mnist_train.csv");
    } else {
        strcpy(fpath,"../../MNIST/mnist_test.csv");
    }

    FILE *file = fopen(fpath, "r");
    float** X_file = malloc(sizeof(float*) * length);
    int* Y_file = malloc(sizeof(int) * length);
    for (int i=0; i < length; i++) {
        X_file[i] = malloc(sizeof(float) * IMAGE_SIZE);
    }


    if (file == NULL) {
        printf("ERROR OPENING FILE\n");
        exit(1);
    }


    char line[MAX_LINE_LENGTH];

    for (int i = 0; i < length; ++i) {
        if (fgets(line, sizeof(line), file) == NULL) {
            printf("ERROR READING LINE %d\n", i);
            exit(1);
        }

        char *token = strtok(line, ",");
        for (int j = 0; j < IMAGE_SIZE; ++j) {
            if (token == NULL) {
                printf("ERROR PARSING IMAGE AT [%d][%d]\n", i, j);
                exit(1);
            }
            X_file[i][j] = atoi(token) / 255.0f;
            token = strtok(NULL, ",");
        }

        if (token == NULL) {
            printf("ERROR PARSING LABEL AT LINE %d\n", i);
            exit(1);
        }
        Y_file[i] = atoi(token);
    }

    *X = X_file;
    *Y = Y_file;
    fclose(file);
}
