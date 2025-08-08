#include <cstdio>
#include <random>
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
    //Applying ReLU
    layer_output[i] = sum + biases[i] > 0 ? sum + biases[i] : 0.0f;
}


float* softmax(float* values, const int size) {
    // Launching a CUDA kernel for a 10-dimension vector would introduce more overhead than performances.
    // Hence, the softmax is not parallelized
    float max_val = values[0];
    for (int i = 1; i < size; i++) {
        if (values[i] > max_val) max_val = values[i];
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        values[i] = expf(values[i] - max_val);
        sum += values[i];
    }
    for (int i = 0; i < size; i++) {
        values[i] /= sum;
    }
    return values;
}



int main() {
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-0.5, 0.5);
    int train_size, test_size, epochs;
    int layer_sizes[MAX_LAYERS];
    int size;

    printf("Reading configurations... \n");
    read_config("../config.txt", &train_size, &test_size, &epochs, layer_sizes, &size);
    printf("Done!\n");
    printf("Network has %i layers\n",size);
    printf("[ ");

    int total_nodes = 0;
    int total_biases = 0;
    int total_weights = 0;

    int* h_bias_indexes = new int[size-1];
    int* h_weights_indexes = new int[size-1];
    int* h_values_indexes = new int[size];
    int* h_deltas_indexes = new int[size];

    h_bias_indexes[0] = 0;
    h_weights_indexes[0] = 0;
    h_values_indexes[0] = 0;
    h_deltas_indexes[0] = 0;

    // Filling the indexes arrays
    for (int i=1; i<size-1; i++) {
        h_weights_indexes[i] = layer_sizes[i-1] * layer_sizes[i] + h_weights_indexes[i-1];
        h_bias_indexes[i] = layer_sizes[i] + h_bias_indexes[i-1];
    }

    for (int i=1; i<size; i++) {
        h_values_indexes[i] = layer_sizes[i-1] + h_values_indexes[i-1];
        h_deltas_indexes[i] = layer_sizes[i-1] + h_values_indexes[i-1];
    }

    for (int i = 0; i < size; i++) {
        total_nodes += layer_sizes[i];
        if (i != 0) total_biases += layer_sizes[i];
        if (i < size - 1) {
            total_weights += layer_sizes[i] * layer_sizes[i+1];
        }
        printf("%i ", layer_sizes[i]);
    }
    printf("] \n");


    double network_size_kb = static_cast<double>(total_weights) * sizeof(float) / 1024.0;
    double dataset_size_mb = (static_cast<double>(train_size) + static_cast<double>(test_size)) * IMAGE_SIZE * sizeof(float) / (1024.0 * 1024.0);
    printf("Network has %i total weights (%.2f KB) and %i total nodes\n", total_weights, network_size_kb, total_nodes);
    printf("The dataset will occupy %.2f MB on the GPU\n", dataset_size_mb);

    float* h_network_values = new float[total_nodes];
    float* h_network_deltas = new float[total_nodes];
    float* h_network_biases = new float[total_biases];
    float* h_network_weights = new float[total_weights];

    // initialize host arrays
    for (int i=0; i < total_nodes; i++) {
        h_network_values[i] = 0.0f;
        h_network_deltas[i] = 0.0f;
    }

    for (int i=0; i < total_biases; i++) {
        h_network_biases[i] = 0.1f;
    }

    for (int i=0; i < total_weights; i++) {
        h_network_weights[i] = distribution(generator);
    }

    //Creating pointers to device memory
    float* d_network_values;
    float* d_network_deltas;
    float* d_network_biases;
    float* d_network_weights;

    // cudaMalloc
    cudaMalloc(&d_network_values, total_nodes * sizeof(float));
    cudaMalloc(&d_network_deltas, total_nodes * sizeof(float));
    cudaMalloc(&d_network_biases, total_biases * sizeof(float));
    cudaMalloc(&d_network_weights, total_weights * sizeof(float));

    // copying from host
    cudaMemcpy(d_network_values, h_network_values, total_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_network_deltas, h_network_deltas, total_nodes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_network_biases, h_network_biases, total_biases * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_network_weights, h_network_weights, total_weights * sizeof(float), cudaMemcpyHostToDevice);


    // Loading the dataset
    float *X_train, *X_test;
    int *Y_train, *Y_test;
    read_dataset(&X_train, &Y_train, true);
    read_dataset(&X_test, &Y_test, false);

    // Creating pointers to device memory
    float* d_X_train;

    //cudaMalloc
    cudaMalloc(&d_X_train, train_size * IMAGE_SIZE * sizeof(float));

    //cudaMemcpy
    cudaMemcpy(d_X_train, X_train, train_size * IMAGE_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("Epoch %d/%d\n", epoch + 1, epochs);
        // For each epoch, cycle on every image
        int correct = 0;
        for (int image_index = 0; image_index < train_size; image_index++) {
            float* d_input_image = d_X_train + image_index * IMAGE_SIZE;
            cudaMemcpy(d_network_values, d_input_image, layer_sizes[0] * sizeof(float), cudaMemcpyDeviceToDevice);

            for (int layer_index = 0; layer_index < size-1; layer_index++) {
                int in_size = layer_sizes[layer_index];
                int out_size = layer_sizes[layer_index+1];

                float* d_layer_input = d_network_values + h_values_indexes[layer_index];
                float* d_layer_output = d_network_values + h_values_indexes[layer_index+1];
                float* d_layer_weights = d_network_weights + h_weights_indexes[layer_index];
                // May seems like an error but it isn't: h_deltas_indexes stores indexes already shifted, so
                // layer_index is ok, layer_index wouldn't
                float* d_layer_biases = d_network_biases + h_deltas_indexes[layer_index];

                int blocks = (out_size + BLK_DIM - 1) / BLK_DIM;
                forwardpass<<<blocks, BLK_DIM>>>(
                    d_layer_output,
                    d_layer_input,
                    d_layer_weights,
                    d_layer_biases,
                    in_size,
                    out_size
                );
                float* h_network_output = new float[layer_sizes[size-1]];
                float* d_network_output = d_network_values + h_values_indexes[size-1];
                cudaMemcpy(h_network_output, d_network_output, layer_sizes[size-1] * sizeof(float), cudaMemcpyDeviceToHost);
                h_network_output = softmax(h_network_output, layer_sizes[size-1]);
                int idx_max = 0;
                for (int i=0; i < layer_sizes[size-1]; i++) {
                    if (h_network_output[i] > h_network_output[idx_max]) idx_max = i;
                }

                if (Y_train[image_index] == idx_max) correct++;
            }

        }
        // Retrieve the last layer values: they should be equal for each epoch (for each run)
        printf("Accuracy: %f\n", static_cast<float>(correct) / static_cast<float>(train_size));

    }






    return 0;
}
