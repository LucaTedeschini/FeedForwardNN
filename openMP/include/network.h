//
// Created by sbrodox on 15/05/25.
//

#ifndef OPENMP_NETWORK_H
#define OPENMP_NETWORK_H


typedef struct {
    int size;
    float* weights;
    float* values;
    float* bias;
    float* delta;
} layer;


#endif //OPENMP_NETWORK_H
