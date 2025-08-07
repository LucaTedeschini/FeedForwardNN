#include <cstdio>
#include "readconfig.hpp"
#include "dataloader.hpp"

#define BLK_DIM 512


__global__ void forwardpass(
    float* layer_output,        //(dim: out_size)
    const float* layer_input,   //(dim: in_size)
    const float* weights,       //(dim: out_size x in_size)
    const float* biases,        //(dim: out_size)
    const int in_size,
    const int out_size
) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= out_size) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < in_size; k++) {
        sum += weights[i * in_size + k] * layer_input[k];
    }
    layer_output[i] = sum + biases[i];
}

__global__ void forwardpass_shared_mem(
    float* layer_output,
    const float* layer_input,
    const float* weights,
    const float* biases,
    int in_size,
    int out_size
) {
    __shared__ float shared_input[BLK_DIM];

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if (i >= out_size) {
        return;
    }

    float sum = 0.0f;

    for (int tile_start = 0; tile_start < in_size; tile_start += BLK_DIM) {
        int input_idx = tile_start + tid;

        if (input_idx < in_size) {
            shared_input[tid] = layer_input[input_idx];
        } else {
            shared_input[tid] = 0.0f; // Padding
        }

        __syncthreads();

        int end_k = tile_start + BLK_DIM;
        for (int k_local = 0; k_local < BLK_DIM && (tile_start + k_local) < in_size; k_local++) {
            sum += weights[i * in_size + (tile_start + k_local)] * shared_input[k_local];
        }

        __syncthreads();
    }

    layer_output[i] = sum + biases[i];
}


int main() {
    int train_size, test_size, epochs;
    int layer_sizes[MAX_LAYERS];
    int size;
    int total_weights = 0, total_nodes = 0;


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

    int* h_value_offsets = new int[size];
    int* h_weight_offsets = new int[size - 1];
    int* h_bias_offsets = new int[size];

    h_value_offsets[0] = 0;
    h_bias_offsets[0] = 0;
    for (int i = 1; i < size; i++) {
        h_value_offsets[i] = h_value_offsets[i-1] + layer_sizes[i-1];
        h_bias_offsets[i] = h_value_offsets[i];
    }

    h_weight_offsets[0] = 0;
    for (int i = 1; i < size - 1; i++) {
        h_weight_offsets[i] = h_weight_offsets[i-1] + layer_sizes[i-1] * layer_sizes[i];
    }

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
    int weights_size = 0;
    for (int i = 0; i<size; i++) {
        floats_required += layer_sizes[i] * 3;
        if (i+1 < size) {
            floats_required += layer_sizes[i] * layer_sizes[i+1];
            weights_size += layer_sizes[i] * layer_sizes[i+1];
        }
    }


    float *X_train, *X_test;
    int *Y_train, *Y_test;

    read_dataset(&X_train, &Y_train, true);
    read_dataset(&X_test, &Y_test, false);


    //Initializing network on host
    float* h_biases = new float[total_nodes];
    float* h_deltas = new float[total_nodes];
    float* h_weights = new float[weights_size];

    for (int i=0; i<total_nodes; i++) {
        h_biases[i] = 0.1f;
        h_deltas[i] = 0.0f;
    }

    for (int i=0; i<weights_size; i++) {
        h_weights[i] = (static_cast<float>(rand()) / static_cast<float>(RAND_MAX)) * 0.02f - 0.01f;
    }


    //Allocate network memory on GPU
    cudaMalloc(&memory_region_network, floats_required * sizeof(float));

    float* d_input = memory_region_network;
    float* d_values = reinterpret_cast<float *>(reinterpret_cast<char *>(memory_region_network) + total_nodes * sizeof(float));
    float* d_biases = reinterpret_cast<float *>(reinterpret_cast<char *>(memory_region_network) + (total_nodes * 2) * sizeof(float));
    float* d_deltas = reinterpret_cast<float *>(reinterpret_cast<char *>(memory_region_network) + (total_nodes * 3) * sizeof(float));
    float* d_weights = reinterpret_cast<float *>(reinterpret_cast<char *>(memory_region_network) + (total_nodes * 4) * sizeof(float));



    //Now we need to allocate the memory for the dataset on the GPU
    float* memory_region_dataset;
    int dataset_float_count = train_size * test_size * IMAGE_SIZE;
    cudaMalloc(&memory_region_dataset, dataset_float_count * sizeof(float));

    float* d_train_X = reinterpret_cast<float *>(reinterpret_cast<char *> (memory_region_dataset));
    float* d_train_Y = reinterpret_cast<float *>(reinterpret_cast<char *> (memory_region_dataset) + train_size * sizeof(float));
    float* d_test_X = reinterpret_cast<float *>(reinterpret_cast<char *> (memory_region_dataset)+ train_size * 2 * sizeof(float));
    float* d_test_Y = reinterpret_cast<float *>(reinterpret_cast<char *> (memory_region_dataset)+ test_size + train_size * 2 * sizeof(float));

    //Copying dataset into GPU memory
    cudaMemcpy(d_train_X, X_train, train_size * IMAGE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_train_Y, Y_train, train_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_X, X_test, test_size * IMAGE_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_Y, Y_test, test_size, cudaMemcpyHostToDevice);

    //Copying network into GPU memory
    cudaMemcpy(d_biases, h_biases, total_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_deltas, h_deltas, total_nodes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_weights, total_weights, cudaMemcpyHostToDevice);


    for (int epoch = 0; epoch < epochs; epoch++) {
        for (int image_index = 0; image_index < train_size; image_index++) {
            // Filling first layer
            float* input_image = &X_train[image_index * IMAGE_SIZE];
            cudaMemcpy(d_values, input_image, IMAGE_SIZE, cudaMemcpyDeviceToDevice);

            for (int layer_index = 0; layer_index < size-1; layer_index++) {
                int in_size = layer_sizes[layer_index];
                int out_size = layer_sizes[layer_index+1];
                float* d_layer_input = d_values + h_value_offsets[layer_index];
                float* d_layer_output = d_values + h_value_offsets[layer_index + 1];
                float* d_layer_weights = d_weights + h_weight_offsets[layer_index];
                float* d_layer_biases = d_biases + h_bias_offsets[layer_index + 1];
                int blocks = (out_size + BLK_DIM - 1) / BLK_DIM;
                forwardpass<<<blocks, BLK_DIM>>>(
                    d_layer_output,
                    d_layer_input,
                    d_layer_weights,
                    d_layer_biases,
                    in_size,
                    out_size
                );
            }
        }


    }

    return 0;
}