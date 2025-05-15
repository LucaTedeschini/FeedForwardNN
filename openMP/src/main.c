#include <stdio.h>

#include "network.h"
#include "utilities.h"

int main(void) {
    int layers_size[] = {2,3,4,5,6};
    int size = 5;
    layer* network = create_network(layers_size, size);

    for (int i=0; i<size-1; i++){
        printf("Layer: %i\n",i);
        for (int j=0; j<network[i].size; j++){
            printf("\tBias: %f\n", network[i].nodes[j].bias);
            for (int z=0; z < network[i+1].size; z++){
                printf("\t\tWeight: %f\n",network[i].nodes[j].weights[z]);
            }
        }
        printf("\n");
    }

    printf("Hello, World!\n");
    return 0;
}
