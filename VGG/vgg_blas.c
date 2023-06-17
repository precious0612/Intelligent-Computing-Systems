#include <cblas.h>

// Assuming row-major order (CBLAS_ORDER = CblasRowMajor)

double*** convolution_cblas(double**** filter, double*** input, double* bias, int stride, int padding, int input_dim, int input_depth, int filter_dim, int num_filters) {
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
                // Create matrices for cblas_dgemm
                double* A = malloc(input_depth * filter_dim * filter_dim * sizeof(double));
                double* B = malloc(input_depth * filter_dim * filter_dim * sizeof(double));

                // Fill A (filter matrix) and B (input matrix)
                for (int d = 0; d < input_depth; d++) {
                    for (int x = 0; x < filter_dim; x++) {
                        for (int y = 0; y < filter_dim; y++) {
                            A[d * filter_dim * filter_dim + x * filter_dim + y] = filter[n][d][x][y];
                            B[d * filter_dim * filter_dim + x * filter_dim + y] = padded_input[d][i * stride + x][j * stride + y];
                        }
                    }
                }

                // Matrix multiplication with cblas_dgemm
                double sum = 0.0;
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, 1, input_depth * filter_dim * filter_dim, 1.0, A, input_depth * filter_dim * filter_dim, B, 1, 0.0, &sum, 1);
                
                output[n][i][j] = sum + bias[n]; // Add bias

                // Free matrices
                free(A);
                free(B);
            }
        }
    }

    free(padded_input);
    return output;
}

