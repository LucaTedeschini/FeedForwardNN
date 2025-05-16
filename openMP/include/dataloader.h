//
// Created by sbrodox on 16/05/25.
//
#ifndef OPENMP_DATALOADER_H
#define OPENMP_DATALOADER_H
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#define MAX_LINE_LENGTH 8000
#define IMAGE_SIZE 784


void read_dataset(float*** X, int** Y, bool is_train);

#endif //OPENMP_DATALOADER_H
