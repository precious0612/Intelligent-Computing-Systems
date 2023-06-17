#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

double*** loadImageAs3DArray(const char* imagePath, int* imgWidth, int* imgHeight, int* imgChannels) {
    int width, height, channels;
    unsigned char *img = stbi_load(imagePath, &width, &height, &channels, 0);
    if(img == NULL) {
        printf("Error loading image.\n");
        return NULL;
    }

    double*** array3D = malloc(channels * sizeof(double**));
    for(int i = 0; i < channels; i++) {
        array3D[i] = malloc(height * sizeof(double*));
        for(int j = 0; j < height; j++) {
            array3D[i][j] = malloc(width * sizeof(double));
        }
    }

    for(int k = 0; k < channels; k++) {
        for(int i = 0; i < height; i++) {
            for(int j = 0; j < width; j++) {
                array3D[k][i][j] = (double) img[(i * width + j) * channels + k];
            }
        }
    }

    *imgWidth = width;
    *imgHeight = height;
    *imgChannels = channels;

    stbi_image_free(img);
    return array3D;
}


double*** pad(double*** input, int padding, int input_dim, int input_depth) {
    int padded_dim = input_dim + 2 * padding;

    // Allocate memory for padded input
    double*** padded_input = malloc(input_depth * sizeof(double**));
    for (int i = 0; i < input_depth; i++) {
        padded_input[i] = malloc(padded_dim * sizeof(double*));
        for (int j = 0; j < padded_dim; j++) {
            padded_input[i][j] = malloc(padded_dim * sizeof(double));
            for (int k = 0; k < padded_dim; k++) {
                // Add padding
                if (j < padding || j >= input_dim + padding || k < padding || k >= input_dim + padding) {
                    padded_input[i][j][k] = 0.0;
                } else {
                    padded_input[i][j][k] = input[i][j - padding][k - padding];
                }
            }
        }
    }

    return padded_input;
}

double*** convolution(double**** filter, double*** input, double* bias, int stride, int padding, int input_dim, int input_depth, int filter_dim, int num_filters) {
    int output_dim = (input_dim - filter_dim + 2 * padding) / stride + 1; // Calculate output dimensions

    // Allocate output array
    double*** output = malloc(num_filters * sizeof(double**));
    for (int i = 0; i < num_filters; i++) {
        output[i] = malloc(output_dim * sizeof(double*));
        for (int j = 0; j < output_dim; j++) {
            output[i][j] = malloc(output_dim * sizeof(double));
        }
    }

    // Pad input
    double*** padded_input = pad(input, padding, input_dim, input_depth);

    // Convolve each filter with input
    for (int n = 0; n < num_filters; n++) {
        for (int i = 0; i < output_dim; i++) {
            for (int j = 0; j < output_dim; j++) {
                double sum = 0.0;
                for (int d = 0; d < input_depth; d++) {
                    for (int x = 0; x < filter_dim; x++) {
                        for (int y = 0; y < filter_dim; y++) {
                            sum += padded_input[d][i * stride + x][j * stride + y] * filter[n][d][x][y];
                        }
                    }
                }
                output[n][i][j] = sum + bias[n]; // Add bias
            }
        }
    }

    free(padded_input);
    return output;
}


double*** max_pooling(double*** input, int pool_size, int stride, int input_dim, int input_depth) {
    // Calculate output dimensions
    int output_dim = (input_dim - pool_size) / stride + 1;

    // Allocate memory for output
    double*** output = malloc(input_depth * sizeof(double**));
    for (int i = 0; i < input_depth; i++) {
        output[i] = malloc(output_dim * sizeof(double*));
        for (int j = 0; j < output_dim; j++) {
            output[i][j] = malloc(output_dim * sizeof(double));
        }
    }

    // Apply max pooling
    for (int i = 0; i < input_depth; i++) {
        for (int j = 0; j < output_dim; j++) {
            for (int k = 0; k < output_dim; k++) {
                double max_val = -INFINITY;
                for (int m = 0; m < pool_size; m++) {
                    for (int n = 0; n < pool_size; n++) {
                        if (input[i][j * stride + m][k * stride + n] > max_val) {
                            max_val = input[i][j * stride + m][k * stride + n];
                        }
                    }
                }
                output[i][j][k] = max_val;
            }
        }
    }

    return output;
}



double**** generateRandom4DData(int dim1, int dim2, int dim3, int dim4) {
    double**** data = malloc(dim1 * sizeof(double***));
    for (int i = 0; i < dim1; i++) {
        data[i] = malloc(dim2 * sizeof(double**));
        for (int j = 0; j < dim2; j++) {
            data[i][j] = malloc(dim3 * sizeof(double*));
            for (int k = 0; k < dim3; k++) {
                data[i][j][k] = malloc(dim4 * sizeof(double));
                for (int l = 0; l < dim4; l++) {
                    data[i][j][k][l] = ((double)rand() / (double)RAND_MAX) * 2 - 1; // random double between -1 and 1
                }
            }
        }
    }
    return data;
}

// Function to free 4D data
void free4DData(double**** data, int dim1, int dim2, int dim3, int dim4) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            for (int k = 0; k < dim3; k++) {
                free(data[i][j][k]);
            }
            free(data[i][j]);
        }
        free(data[i]);
    }
    free(data);
}

// Function to generate 3D array with random data
double*** generateRandom3DData(int dim1, int dim2, int dim3) {
    double*** data = malloc(dim1 * sizeof(double**));
    for (int i = 0; i < dim1; i++) {
        data[i] = malloc(dim2 * sizeof(double*));
        for (int j = 0; j < dim2; j++) {
            data[i][j] = malloc(dim3 * sizeof(double));
            for (int k = 0; k < dim3; k++) {
                data[i][j][k] = ((double)rand() / (double)RAND_MAX) * 2 - 1; // random double between -1 and 1
            }
        }
    }
    return data;
}

// Function to free 3D data
void free3DData(double*** data, int dim1, int dim2, int dim3) {
    for (int i = 0; i < dim1; i++) {
        for (int j = 0; j < dim2; j++) {
            free(data[i][j]);
        }
        free(data[i]);
    }
    free(data);
}

double** generateRandom2DData(int rows, int cols) {
    double** array = (double**)malloc(rows * sizeof(double*));
    for(int i = 0; i < rows; i++) {
        array[i] = (double*)malloc(cols * sizeof(double));
        for(int j = 0; j < cols; j++) {
            array[i][j] = (double)rand() / RAND_MAX; // Generates a random double between 0 and 1
        }
    }
    return array;
}

// Function to free 2D data
void free2DData(double** data, int rows, int cols) {
    for(int i = 0; i < rows; i++) {
        free(data[i]);
    }
    free(data);
}


// Function to generate 1D array with random data
double* generateRandom1DData(int dim) {
    double* data = malloc(dim * sizeof(double));
    for (int i = 0; i < dim; i++) {
        data[i] = ((double)rand() / (double)RAND_MAX) * 2 - 1; // random double between -1 and 1
    }
    return data;
}

// Function to free 1D data
void free1DData(double* data) {
    free(data);
}

// Function to flatten a 3D array
double* flatten(double*** input, int input_dim, int input_depth) {
    double* flat_input = malloc(input_dim * input_dim * input_depth * sizeof(double));
    for (int i = 0; i < input_depth; i++) {
        for (int j = 0; j < input_dim; j++) {
            for (int k = 0; k < input_dim; k++) {
                flat_input[i * input_dim * input_dim + j * input_dim + k] = input[i][j][k];
            }
        }
    }
    return flat_input;
}

// Function to perform full connection
double* fully_connected(double* flat_input, double** weights, double* biases, int input_dim, int output_dim) {
    // Multiply weights and add biases
    double* output = malloc(output_dim * sizeof(double));
    for (int i = 0; i < output_dim; i++) {
        double sum = 0.0;
        for (int j = 0; j < input_dim; j++) {
            sum += flat_input[j] * weights[i][j];
        }
        output[i] = sum + biases[i];
    }

    return output;
}

// ReLU for 3D matrix
void relu_3d(double*** input, int dim1, int dim2, int dim3) {
    if (input == NULL) {
        printf("Invalid input for relu_3d function\n");
        return;
    }
    
    for (int i = 0; i < dim1; i++) {
        if (input[i] == NULL) {
            printf("Invalid input at index %d for relu_3d function\n", i);
            return;
        }
        
        for (int j = 0; j < dim2; j++) {
            if (input[i][j] == NULL) {
                printf("Invalid input at index %d, %d for relu_3d function\n", i, j);
                return;
            }
            
            for (int k = 0; k < dim3; k++) {
                if (input[i][j][k] < 0) {
                    input[i][j][k] = 0;
                }
            }
        }
    }
}


// Softmax for vector
void softmax_vector(double* input, int length) {
    // Calculate exponent of each element in the input
    double max_val = input[0];
    for (int i = 1; i < length; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    // Subtract max_val to avoid overflow
    double sum = 0.0;
    for (int i = 0; i < length; i++) {
        input[i] = exp(input[i] - max_val);
        sum += input[i];
    }

    // Normalize so that the input sums to 1
    for (int i = 0; i < length; i++) {
        input[i] /= sum;
    }
}

// ReLU for vector
void relu_vector(double* input, int length) {
    for (int i = 0; i < length; i++) {
        if (input[i] < 0) {
            input[i] = 0;
        }
    }
}

int main() {
    srand(time(NULL)); // Seed the random number generator

    // Step 1: Random initialization
    // Suppose our input image is of size 224x224x3
    // double*** input_img = generateRandom3DData(3, 224, 224);
    int imgChannels = 3;
    int imgHeight = 224;
    int imgWidth = 224;
    double*** input_img = loadImageAs3DArray("/Users/precious/Intelligent-Computing-Systems.git/VGG/Beautiful_demoiselle_(Calopteryx_virgo)_male_3.jpg", &imgChannels, &imgWidth, &imgHeight);


    // For the first Conv3D layer, we have 64 filters of size 3x3x3
    double**** conv1_weights = generateRandom4DData(64, 3, 3, 3);
    double* conv1_biases = generateRandom1DData(64);
    double**** conv2_weights = generateRandom4DData(64, 64, 3, 3);
    double* conv2_biases = generateRandom1DData(64);
    double**** conv3_weights = generateRandom4DData(128, 64, 3, 3);
    double* conv3_biases = generateRandom1DData(128);
    double**** conv4_weights = generateRandom4DData(128, 128, 3, 3);
    double* conv4_biases = generateRandom1DData(128);
    double**** conv5_weights = generateRandom4DData(256, 128, 3, 3);
    double* conv5_biases = generateRandom1DData(256);
    double**** conv6_weights = generateRandom4DData(256, 256, 3, 3);
    double* conv6_biases = generateRandom1DData(256);
    double**** conv7_weights = generateRandom4DData(256, 256, 3, 3);
    double* conv7_biases = generateRandom1DData(256);
    double**** conv8_weights = generateRandom4DData(256, 256, 3, 3);
    double* conv8_biases = generateRandom1DData(256);
    double**** conv9_weights = generateRandom4DData(512, 256, 3, 3);
    double* conv9_biases = generateRandom1DData(512);
    double**** conv10_weights = generateRandom4DData(512, 512, 3, 3);
    double* conv10_biases = generateRandom1DData(512);
    double**** conv11_weights = generateRandom4DData(512, 512, 3, 3);
    double* conv11_biases = generateRandom1DData(512);
    double**** conv12_weights = generateRandom4DData(512, 512, 3, 3);
    double* conv12_biases = generateRandom1DData(512);
    double**** conv13_weights = generateRandom4DData(512, 512, 3, 3);
    double* conv13_biases = generateRandom1DData(512);
    double**** conv14_weights = generateRandom4DData(512, 512, 3, 3);
    double* conv14_biases = generateRandom1DData(512);
    double**** conv15_weights = generateRandom4DData(512, 512, 3, 3);
    double* conv15_biases = generateRandom1DData(512);
    double**** conv16_weights = generateRandom4DData(512, 512, 3, 3);
    double* conv16_biases = generateRandom1DData(512);
    double* fc1_weights = generateRandom2DData(25088, 4096);
    double* fc1_biases = generateRandom1DData(4096);
    double* fc2_weights = generateRandom2DData(4096, 4096);
    double* fc2_biases = generateRandom1DData(4096);
    double* fc3_weights = generateRandom2DData(4096, 1000);
    double* fc3_biases = generateRandom1DData(1000);

    // Forward pass
    clock_t start_time = clock(); // Record the start time

    // Step 2: Convolution
    // convolution(double**** filter, double*** input, double* bias, int stride, int padding, int input_dim, int input_depth, int filter_dim, int num_filters)
    double*** conv1_out = convolution(conv1_weights, input_img, conv1_biases, 1, 1, 224, 3, 3, 64);
    relu_3d(conv1_out, 64, 224, 224);
    double*** conv2_out = convolution(conv2_weights, conv1_out, conv2_biases, 1, 1, 224, 3, 3, 64);
    relu_3d(conv2_out, 64, 224, 224);
    // max_pooling(double*** input, int pool_size, int stride, int input_dim, int input_depth) 
    double*** pool1_out = max_pooling(conv2_out, 2, 2, 224, 64);
    double*** conv3_out = convolution(conv3_weights, pool1_out, conv3_biases, 1, 1, 112, 3, 3, 128);
    relu_3d(conv3_out, 128, 112, 112);
    double*** conv4_out = convolution(conv4_weights, conv3_out, conv4_biases, 1, 1, 112, 3, 3, 128);
    relu_3d(conv4_out, 128, 112, 112);
    double*** pool2_out = max_pooling(conv4_out, 2, 2, 112, 128);
    double*** conv5_out = convolution(conv5_weights, pool2_out, conv5_biases, 1, 1, 56, 3, 3, 256);
    relu_3d(conv5_out, 256, 56, 56);
    double*** conv6_out = convolution(conv6_weights, conv5_out, conv6_biases, 1, 1, 56, 3, 3, 256);
    relu_3d(conv6_out, 256, 56, 56);
    double*** conv7_out = convolution(conv7_weights, conv6_out, conv7_biases, 1, 1, 56, 3, 3, 256);
    relu_3d(conv7_out, 256, 56, 56);
    double*** conv8_out = convolution(conv8_weights, conv7_out, conv8_biases, 1, 1, 56, 3, 3, 256);
    relu_3d(conv8_out, 256, 56, 56);
    double*** pool3_out = max_pooling(conv8_out, 2, 2, 56, 256);
    double*** conv9_out = convolution(conv9_weights, pool3_out, conv9_biases, 1, 1, 28, 3, 3, 512);
    relu_3d(conv9_out, 512, 28, 28);
    double*** conv10_out = convolution(conv10_weights, conv9_out, conv10_biases, 1, 1, 28, 3, 3, 512);
    relu_3d(conv10_out, 512, 28, 28);
    double*** conv11_out = convolution(conv11_weights, conv10_out, conv11_biases, 1, 1, 28, 3, 3, 512);
    relu_3d(conv11_out, 512, 28, 28);
    double*** conv12_out = convolution(conv12_weights, conv11_out, conv12_biases, 1, 1, 28, 3, 3, 512);
    relu_3d(conv12_out, 512, 28, 28);
    double*** pool4_out = max_pooling(conv12_out, 2, 2, 28, 512);
    double*** conv13_out = convolution(conv13_weights, pool4_out, conv13_biases, 1, 1, 14, 3, 3, 512);
    relu_3d(conv13_out, 512, 14, 14);
    double*** conv14_out = convolution(conv14_weights, conv13_out, conv14_biases, 1, 1, 14, 3, 3, 512);
    relu_3d(conv14_out, 512, 14, 14);
    double*** conv15_out = convolution(conv15_weights, conv14_out, conv15_biases, 1, 1, 14, 3, 3, 512);
    relu_3d(conv15_out, 512, 14, 14);
    double*** conv16_out = convolution(conv16_weights, conv15_out, conv16_biases, 1, 1, 14, 3, 3, 512);
    relu_3d(conv16_out, 512, 14, 14);
    double*** pool5_out = max_pooling(conv16_out, 2, 2, 14, 512);

    // Step 3: Fully Connected
    double* flatten_out = flatten(pool5_out, 7, 512);
    double* fc1_out = fully_connected(flatten_out, fc1_weights, fc1_biases, 25088, 4096);
    relu_vector(fc1_out, 4096); 
    double* fc2_out = fully_connected(fc1_out, fc2_weights, fc2_biases, 4096, 4096);
    relu_vector(fc2_out, 4096);
    double* fc3_out = fully_connected(fc2_out, fc3_weights, fc3_biases, 4096, 1000);
    softmax_vector(fc3_out, 1000);

    clock_t end_time = clock(); // Record the end time
    double total_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    // Print the results
    // printf("fc3_out[0]: %f\n", fc3_out[0]);
    printf("Total time: %f\n", total_time);

    // Free the memory
    free4DData(conv1_weights, 64, 3, 3, 3);
    free1DData(conv1_biases);
    free4DData(conv2_weights, 64, 64, 3, 3);
    free1DData(conv2_biases);
    free4DData(conv3_weights, 128, 64, 3, 3);
    free1DData(conv3_biases);
    free4DData(conv4_weights, 128, 128, 3, 3);
    free1DData(conv4_biases);
    free4DData(conv5_weights, 256, 128, 3, 3);
    free1DData(conv5_biases);
    free4DData(conv6_weights, 256, 256, 3, 3);
    free1DData(conv6_biases);
    free4DData(conv7_weights, 256, 256, 3, 3);
    free1DData(conv7_biases);
    free4DData(conv8_weights, 256, 256, 3, 3);
    free1DData(conv8_biases);
    free4DData(conv9_weights, 512, 256, 3, 3);
    free1DData(conv9_biases);
    free4DData(conv10_weights, 512, 512, 3, 3);
    free1DData(conv10_biases);
    free4DData(conv11_weights, 512, 512, 3, 3);
    free1DData(conv11_biases);
    free4DData(conv12_weights, 512, 512, 3, 3);
    free1DData(conv12_biases);
    free4DData(conv13_weights, 512, 512, 3, 3);
    free1DData(conv13_biases);
    free4DData(conv14_weights, 512, 512, 3, 3);
    free1DData(conv14_biases);
    free4DData(conv15_weights, 512, 512, 3, 3);
    free1DData(conv15_biases);
    free4DData(conv16_weights, 512, 512, 3, 3);
    free1DData(conv16_biases);

    free3DData(input_img, 3, 224, 224);
    free3DData(conv1_out, 64, 224, 244);
    free3DData(conv2_out, 64, 224, 244);
    free3DData(pool1_out, 64, 112, 112);
    free3DData(conv3_out, 128, 112, 112);
    free3DData(conv4_out, 128, 112, 112);
    free3DData(pool2_out, 128, 56, 56);
    free3DData(conv5_out, 256, 56, 56);
    free3DData(conv6_out, 256, 56, 56);
    free3DData(conv7_out, 256, 56, 56);
    free3DData(conv8_out, 256, 56, 56);
    free3DData(pool3_out, 256, 28, 28);
    free3DData(conv9_out, 512, 28, 28);
    free3DData(conv10_out, 512, 28, 28);
    free3DData(conv11_out, 512, 28, 28);
    free3DData(conv12_out, 512, 28, 28);
    free3DData(pool4_out, 512, 14, 14);
    free3DData(conv13_out, 512, 14, 14);
    free3DData(conv14_out, 512, 14, 14);
    free3DData(conv15_out, 512, 14, 14);
    free3DData(conv16_out, 512, 14, 14);
    free3DData(pool5_out, 512, 7, 7);
    free1DData(flatten_out);
    free1DData(fc1_out);
    free1DData(fc2_out);
    free1DData(fc3_out);

    return 0;
}
