#include <cstdio>
#include "readconfig.hpp"
#include "dataloader.hpp"

int main() {
    int train_size, test_size, epochs;
    int layer_sizes[MAX_LAYERS];
    int size;
    int total_weights = 0, total_nodes = 0;
    double start, end;

    printf("Reading configurations... \n");
    read_config("../config.txt", &train_size, &test_size, &epochs, layer_sizes, &size);
    printf("Done!\n");
    printf("Network has %i layers\n",size);
    printf("[ ");
    for (int i = 0; i<size; i++) {
        total_nodes+=layer_sizes[i];
        if (i < size-1) total_weights += layer_sizes[i] * layer_sizes[i+1];
        printf("%i ",layer_sizes[i]);
    }
    printf("] \n");


    double network_size = static_cast<double>(total_weights)*4/1024;
    double dataset_size = (static_cast<double>(train_size) + static_cast<double>(test_size)) * IMAGE_SIZE * 2 / 1024 / 1024;

    printf("Network has %i total weights (%fKB) and %i total nodes\n",total_weights,network_size,total_nodes);
    printf("The dataset will occupy %fMB on the GPU\n", dataset_size);
    printf("Total weight in GPU memory: %fMB\n", dataset_size + network_size / 1024);

    // Reserving the space on the Device
    float* memory_region_network;

    // Total memory is given by the sum of each layer's values, weights, bias and deltas
    // Total input size = layer_sizes[0]
    // Total value = sum(layer_sizes[i])
    // Total biases = sum(layer_sizes[i])
    // Total deltas = sum(layer_sizes[i])
    // Total weights = sum(layer_sizes[i] * layer_sizes[i+1])
    int floats_required = layer_sizes[0];
    for (int i = 0; i<size; i++) {
        floats_required += layer_sizes[i] * 3;
        if (i+1 < size) floats_required += layer_sizes[i] * layer_sizes[i+1];
    }

    cudaMalloc(&memory_region_network, floats_required * sizeof(float));

    float* d_input = memory_region_network;
    float* d_values = reinterpret_cast<float *>(reinterpret_cast<char *>(memory_region_network) + total_nodes * sizeof(float));
    float* d_biases = reinterpret_cast<float *>(reinterpret_cast<char *>(memory_region_network) + (total_nodes * 2) * sizeof(float));
    float* d_deltas = reinterpret_cast<float *>(reinterpret_cast<char *>(memory_region_network) + (total_nodes * 3) * sizeof(float));
    float* d_weights = reinterpret_cast<float *>(reinterpret_cast<char *>(memory_region_network) + (total_nodes * 4) * sizeof(float));


    float **X_train, **X_test;
    int *Y_train, *Y_test;

    read_dataset(&X_train, &Y_train, true);
    read_dataset(&X_test, &Y_test, false);

    //Now we need to allocate the memory for the dataset on the GPU
    float* memory_region_dataset;
    int dataset_float_count = train_size * test_size * IMAGE_SIZE;
    cudaMalloc(&memory_region_dataset, dataset_float_count * sizeof(float));


    return 0;
}