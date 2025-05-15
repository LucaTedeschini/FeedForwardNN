//
// Created by sbrodox on 15/05/25.
//

#ifndef OPENMP_NETWORK_H
#define OPENMP_NETWORK_H

typedef struct {
    float* weights;
    float bias;
    float value;
    float delta;
} node;

typedef struct {
    node* nodes;
    int size;
} layer;


#endif //OPENMP_NETWORK_H
