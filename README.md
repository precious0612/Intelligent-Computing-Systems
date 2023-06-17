# Intelligent Computing System Course Repository

This repository includes all the code and documentation related to the Intelligent Computing System course I'm taking. The main project is the implementation of the VGG19 network, first with numpy, and then with C/C++, with an emphasis on optimizing the convolution operation, which is the most time-consuming operation in the network.

## Project Structure

The main files in this repository are:

- `Convolution Calculation/test.ipynb`: This is where I initially implemented the convolution operation using numpy.
- `Convolution Calculation/conv.py or conv_class.py`: After implementing the convolution operation, I used numpy to replicate the VGG19 network for a specific task.
- `VGG/vgg19.c` and `VGG/vgg.cpp`: When I found that the inference speed of my network was significantly slower than that of networks running on frameworks such as TensorFlow or PyTorch, I decided to use C/C++ to replicate the VGG19 network.
- `VGG19/*`: These files are part of a project for a school competition where the goal was to implement a 1D convolution in C and then optimize it.

## Optimization Journey

At first, I found the speed of my network to be much slower than that of popular frameworks such as TensorFlow or PyTorch. After researching the issue, I learned that these frameworks make extensive use of C/C++ interfaces at a lower level. This inspired me to try implementing the VGG19 network in C/C++.

In particular, the convolution operation is the most time-consuming operation in the network. Therefore, I spent a significant amount of time optimizing this operation in various ways:

### Original Convolution Operation

The initial implementation of the convolution operation was quite straightforward. However, it had issues with memory fragmentation which could lead to inefficient memory usage and slow down the execution. 

```c
double*** convolution1d(double**** filter, double*** input, double* bias, int stride, int padding, int input_dim, int input_depth, int filter_dim, int num_filters) {
    int output_dim = (input_dim - filter_dim + 2 * padding) / stride + 1; // Calculate output dimensions

    // Allocate output array
    double*** output = malloc(num_filters * sizeof(double**));
    for (int i = 0; i < num_filters; i++) {
        output[i] = malloc(output_dim * sizeof(double*));
        for (int j = 0; j < output_dim; j++) {
            output[i][j] = malloc(1 * sizeof(double));
        }
    }

    // Convolve each filter with input
    for (int n = 0; n < num_filters; n++) {
        for (int i = 0; i < output_dim; i++) {
            int j = 0;
            double sum = 0.0;
            for (int d = 0; d < input_depth; d++) {
                for (int x = 0; x < filter_dim; x++) {
                    sum += input[d][i * stride + x][j] * filter[n][d][x][0];
                }
            }
            output[n][i][j] = sum + bias[n]; // Add bias
        }
    }

    return output;
}
```

### Avoiding Memory Fragmentation

To avoid memory fragmentation, I allocated the output array in a continuous block of memory. This modification resulted in a significant increase in speed.

```c
double*** convolution1d(double**** filter, double*** input, double* bias, int stride, int padding, int input_dim, int input_depth, int filter_dim, int num_filters) {
    int output_dim = (input_dim - filter_dim + 2 * padding) / stride + 1; // Calculate output dimensions

    // Allocate output array in a continuous block of memory
    double*** output = malloc(num_filters * sizeof(double**));
    double* output_data = malloc(num_filters * output_dim * sizeof(double));

    for (int i = 0; i < num_filters; i++) {
        output[i] = malloc(output_dim * sizeof(double*));
        for (int j = 0; j < output_dim; j++) {
            output[i][j] = &output_data[i * output_dim + j];
        }
    }

    // Convolve each filter with input
    for (int n = 0; n < num_filters; n++) {
        for (int i = 0; i < output_dim; i++) {
            int j = 0;
            double sum = 0.0;
            for (int d = 0; d < input_depth; d++) {
                for (int x = 0; x < filter_dim; x++) {
                    sum += input[d][i * stride + x][j] * filter[n][d][x][0];
                }
            }
            output[n][i][j] = sum + bias[n]; // Add bias
        }
    }

    return output;
}
```

### Loop Unrolling

To further speed up the convolution operation, I applied the technique of loop unrolling. Loop unrolling is a simple optimization technique where the number of iterations in a loop is increased to reduce the overhead of loop control. 

```c
double*** convolution1d(double**** filter, double*** input, double* bias, int stride, int padding, int input_dim, int input_depth, int filter_dim, int num_filters) {
    int output_dim = (input_dim - filter_dim + 2 * padding) / stride + 1; // Calculate output dimensions

    // Allocate output array in a continuous block of memory
    double*** output = malloc(num_filters * sizeof(double**));
    double* output_data = malloc(num_filters * output_dim * sizeof(double));

    for (int i = 0; i < num_filters; i++) {
        output[i] = malloc(output_dim * sizeof(double*));
        for (int j = 0; j < output_dim; j++) {
            output[i][j] = &output_data[i * output_dim + j];
        }
    }

    // Convolve each filter with input
    for (int n = 0; n < num_filters; n++) {
        for (int i = 0; i < output_dim; i++) {
            int j = 0;
            double sum = 0.0;
            for (int d = 0; d < input_depth; d++) {
                // Handle all pairs of elements
                for (int x = 0; x < filter_dim - 1; x += 2) {
                    sum += input[d][i * stride + x][j] * filter[n][d][x][0];
                    sum += input[d][i * stride + x + 1][j] * filter[n][d][x + 1][0];
                }

                // Handle the last element if filter_dim is odd
                if (filter_dim % 2 != 0) {
                    sum += input[d][i * stride + filter_dim - 1][j] * filter[n][d][filter_dim - 1][0];
                }
            }
            output[n][i][j] = sum + bias[n]; // Add bias
        }
    }

    free(output_data);

    return output;
}
```

### Reducing Unnecessary Memory Allocation

In the final version of the code, I further reduced unnecessary memory allocation by removing the innermost dimension from the output array. This change decreased the number of memory allocations, reducing the time spent in memory management and improving performance. 

```c
double** convolution1d(double**** filter, double*** input, double* bias, int stride, int padding, int input_dim, int input_depth, int filter_dim, int num_filters) {
    int output_dim = (input_dim - filter_dim + 2 * padding) / stride + 1; // Calculate output dimensions

    // Allocate output array in a continuous block of memory
    double** output = malloc(num_filters * sizeof(double*));
    double* output_data = malloc(num_filters * output_dim * sizeof(double));

    for (int i = 0; i < num_filters; i++) {
        output[i] = &output_data[i * output_dim];
    }

    // Convolve each filter with input
    for (int n = 0; n < num_filters; n++) {
        for (int i = 0; i < output_dim; i++) {
            double sum = 0.0;
            for (int d = 0; d < input_depth; d++) {
                // Handle all pairs of elements
                for (int x = 0; x < filter_dim - 1; x += 2) {
                    sum += input[d][i * stride + x][0] * filter[n][d][x][0];
                    sum += input[d][i * stride + x + 1][0] * filter[n][d][x + 1][0];
                }

                // Handle the last element if filter_dim is odd
                if (filter_dim % 2 != 0) {
                    sum += input[d][i * stride + filter_dim - 1][0] * filter[n][d][filter_dim - 1][0];
                }
            }
            output[n][i] = sum + bias[n]; // Add bias
        }
    }

    free(output_data);

    return output;
}
```

The actual implementation slightly differs from the above code. For the exact implementation, please refer to the `vgg19_1d.c` file.

## Conclusion

This repository represents my journey in learning about neural networks and how to optimize them for speed. I hope this can be a useful resource for others who are interested in neural network implementation and optimization.

Note: For a more detailed explanation of the code and the optimization techniques used, please refer to the individual comments in the code files.
