#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <chrono>
#include <typeinfo>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

std::vector<std::vector<std::vector<float>>> loadImageData(const std::string& image_path) {
    // Load the image
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "Could not open or find the image\n";
        return {};
    }

    // Resize the image to 224x224
    cv::resize(img, img, cv::Size(224, 224));

    // Convert the image from BGR to RGB
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);

    // Convert the image from 8-bit integers to double precision
    img.convertTo(img, CV_64F);

    // Normalize the image to be between -1 and 1
    img = img / 127.5 - 1;

    // Convert the Mat image to 3D array
    std::vector<std::vector<std::vector<float>>> input_img(3, std::vector<std::vector<float>>(224, std::vector<float>(224)));
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            cv::Vec3d pixel = img.at<cv::Vec3d>(i, j);
            for (int k = 0; k < 3; k++) {
                input_img[k][i][j] = pixel[k];
            }
        }
    }

    return input_img;
}

std::vector<std::vector<std::vector<float>>> convolution(
    const std::vector<std::vector<std::vector<float>>>& input, // [channels][height][width]
    const std::vector<std::vector<std::vector<std::vector<float>>>>& filters, // [out_channels][in_channels][filter_height][filter_width]
    int stride, 
    int padding
) {
    int inChannels = input.size();
    int inHeight = input[0].size();
    int inWidth = input[0][0].size();

    int outChannels = filters.size();
    int filterHeight = filters[0][0].size();
    int filterWidth = filters[0][0][0].size();

    int outHeight = (inHeight - filterHeight + 2 * padding) / stride + 1;
    int outWidth = (inWidth - filterWidth + 2 * padding) / stride + 1;

    std::vector<std::vector<std::vector<float>>> output(outChannels, std::vector<std::vector<float>>(outHeight, std::vector<float>(outWidth)));

    for (int o = 0; o < outChannels; ++o) {
        for (int i = 0; i < outHeight; ++i) {
            for (int j = 0; j < outWidth; ++j) {
                float sum = 0.0f;
                for (int c = 0; c < inChannels; ++c) {
                    for (int h = 0; h < filterHeight; ++h) {
                        for (int w = 0; w < filterWidth; ++w) {
                            int hOffset = i * stride - padding + h;
                            int wOffset = j * stride - padding + w;
                            if (hOffset >= 0 && hOffset < inHeight && wOffset >= 0 && wOffset < inWidth) {
                                sum += filters[o][c][h][w] * input[c][hOffset][wOffset];
                            }
                        }
                    }
                }
                output[o][i][j] = sum;
            }
        }
    }

    return output;
}

std::vector<std::vector<std::vector<float>>> max_pooling(
    const std::vector<std::vector<std::vector<float>>>& input, // [channels][height][width]
    int poolSize, 
    int stride
) {
    int channels = input.size();
    int inHeight = input[0].size();
    int inWidth = input[0][0].size();

    int outHeight = (inHeight - poolSize) / stride + 1;
    int outWidth = (inWidth - poolSize) / stride + 1;

    std::vector<std::vector<std::vector<float>>> output(channels, std::vector<std::vector<float>>(outHeight, std::vector<float>(outWidth)));

    for (int c = 0; c < channels; ++c) {
        for (int i = 0; i < outHeight; ++i) {
            for (int j = 0; j < outWidth; ++j) {
                float maxVal = std::numeric_limits<float>::lowest();
                for (int h = 0; h < poolSize; ++h) {
                    for (int w = 0; w < poolSize; ++w) {
                        int hOffset = i * stride + h;
                        int wOffset = j * stride + w;
                        maxVal = std::max(maxVal, input[c][hOffset][wOffset]);
                    }
                }
                output[c][i][j] = maxVal;
            }
        }
    }

    return output;
}

std::vector<float> flatten(const std::vector<std::vector<std::vector<float>>>& input) {
    std::vector<float> output;
    for (const auto& channel : input) {
        for (const auto& row : channel) {
            for (float val : row) {
                output.push_back(val);
            }
        }
    }
    return output;
}

std::vector<float> fully_connected(const std::vector<float>& input, const std::vector<std::vector<float>>& weights, const std::vector<float>& biases) {
    std::vector<float> output(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
        output[i] = biases[i];
        for (size_t j = 0; j < input.size(); ++j) {
            output[i] += weights[i][j] * input[j];
        }
    }
    return output;
}

void relu(std::vector<float>& input) {
    for (float& val : input) {
        val = std::max(0.0f, val);
    }
}

void softmax(std::vector<float>& input) {
    float maxVal = *std::max_element(input.begin(), input.end());
    float sum = 0.0f;

    for (float& val : input) {
        val = std::exp(val - maxVal); // Subtract maxVal for numerical stability
        sum += val;
    }

    for (float& val : input) {
        val /= sum;
    }
}

class Layer {
public:
    virtual ~Layer() = default; // virtual destructor

    // Process 3D input through the layer and return output.
    // This will be used by Conv2DLayer.
    virtual std::vector<std::vector<std::vector<float>>> forward3D(const std::vector<std::vector<std::vector<float>>>& input) {
        throw std::logic_error("This layer does not support 3D input.");
    }

    // Process 1D input through the layer and return output.
    // This will be used by FullyConnectedLayer.
    virtual std::vector<float> forward1D(const std::vector<float>& input) {
        throw std::logic_error("This layer does not support 1D input.");
    }

    //flatten
    virtual std::vector<float> forwardFlatten(const std::vector<std::vector<std::vector<float>>>& input) {
        throw std::logic_error("This layer does not support flatten.");
    }
};


class ConvolutionalLayer : public Layer {
public:
    ConvolutionalLayer(const std::vector<std::vector<std::vector<std::vector<float>>>>& filters, int stride, int padding)
        : filters(filters), stride(stride), padding(padding) {}

    std::vector<std::vector<std::vector<float>>> forward3D(const std::vector<std::vector<std::vector<float>>>& input) override {
        return convolution(input, filters, stride, padding);
    }
private:
    std::vector<std::vector<std::vector<std::vector<float>>>> filters;
    int stride;
    int padding;
};

class MaxPoolingLayer : public Layer {
public:
    MaxPoolingLayer(int poolSize, int stride)
        : poolSize(poolSize), stride(stride) {}

    std::vector<std::vector<std::vector<float>>> forward3D(const std::vector<std::vector<std::vector<float>>>& input) override {
        return max_pooling(input, poolSize, stride);
    }

private:
    int poolSize;
    int stride;
};

// ReLU activation layer class
class ReLUActivationLayer : public Layer {
public:
    std::vector<std::vector<std::vector<float>>> forward3D(const std::vector<std::vector<std::vector<float>>>& input) override {
        std::vector<std::vector<std::vector<float>>> output = input;
        for (auto& channel : output) {
            for (auto& row : channel) {
                relu(row); // use the relu function defined before
            }
        }
        return output;
    }

private:
    // Extend relu function to handle 1D vector
    void relu(std::vector<float>& input) {
        for (float& val : input) {
            val = std::max(0.0f, val);
        }
    }
};

// Flatten layer class
class FlattenLayer : public Layer {
public:
    std::vector<float> forwardFlatten(const std::vector<std::vector<std::vector<float>>>& input) override {
        return flatten(input);
    }
};

// Fully connected layer class
class FullyConnectedLayer : public Layer {
public:
    FullyConnectedLayer(const std::vector<std::vector<float>>& weights, const std::vector<float>& biases)
        : weights(weights), biases(biases) {}

    std::vector<float> forward1D(const std::vector<float>& input) override {
        return fully_connected(input, weights, biases);
    }

private:
    std::vector<std::vector<float>> weights;
    std::vector<float> biases;
};

// ReLU activation layer class for fully connected layer
class FCReLUActivationLayer : public Layer {
public:
    std::vector<float> forward1D(const std::vector<float>& input) override {
        std::vector<float> output = input;
        relu(output);
        return output;
    }

private:
    void relu(std::vector<float>& input) {
        for (float& val : input) {
            val = std::max(0.0f, val);
        }
    }
};

// Softmax activation layer class
class SoftmaxActivationLayer : public Layer {
public:
    std::vector<float> forward1D(const std::vector<float>& input) override {
        return softmax(input);
    }

private:
    std::vector<float> softmax(const std::vector<float>& input) {
        std::vector<float> output = input;
        float maxVal = *max_element(output.begin(), output.end());
        float sum = 0.0f;
        
        // Subtract max for numerical stability
        for (float& val : output) {
            val = exp(val - maxVal);
            sum += val;
        }

        // Normalize
        for (float& val : output) {
            val /= sum;
        }

        return output;
    }
};


// Generate random filters for convolution
std::vector<std::vector<std::vector<std::vector<float>>>> generateRandomFilters(int outChannels, int inChannels, int filterHeight, int filterWidth) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<std::vector<std::vector<std::vector<float>>>> filters(outChannels, std::vector<std::vector<std::vector<float>>>(inChannels, std::vector<std::vector<float>>(filterHeight, std::vector<float>(filterWidth))));

    for (int o = 0; o < outChannels; ++o) {
        for (int i = 0; i < inChannels; ++i) {
            for (int h = 0; h < filterHeight; ++h) {
                for (int w = 0; w < filterWidth; ++w) {
                    filters[o][i][h][w] = dis(gen);
                }
            }
        }
    }

    return filters;
}

// Function to generate random data
std::vector<std::vector<std::vector<float>>> generateRandom3DData(int channels, int height, int width) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<std::vector<std::vector<float>>> data(channels, std::vector<std::vector<float>>(height, std::vector<float>(width)));

    for (int c = 0; c < channels; ++c) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                data[c][h][w] = dis(gen);
            }
        }
    }

    return data;
}

std::vector<std::vector<float>> generateRandom2DData(int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<std::vector<float>> data(rows, std::vector<float>(cols));

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i][j] = dis(gen);
        }
    }

    return data;
}

// Function to generate random 1D data
std::vector<float> generateRandom1DData(int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<float> data(size);

    for (int i = 0; i < size; ++i) {
        data[i] = dis(gen);
    }

    return data;
}

// VGG19 network architecture
class ConvNet {
public:
    ConvNet() {
        // Initialize layers for VGG19
        // Convolutional layers
        layers.push_back(new ConvolutionalLayer(generateRandomFilters(64, 3, 3, 3), 1, 1));
        layers.push_back(new ReLUActivationLayer());
        layers.push_back(new ConvolutionalLayer(generateRandomFilters(64, 64, 3, 3), 1, 1));
        layers.push_back(new ReLUActivationLayer());
        layers.push_back(new MaxPoolingLayer(2, 2));

        layers.push_back(new ConvolutionalLayer(generateRandomFilters(128, 64, 3, 3), 1, 1));
        layers.push_back(new ReLUActivationLayer());
        layers.push_back(new ConvolutionalLayer(generateRandomFilters(128, 128, 3, 3), 1, 1));
        layers.push_back(new ReLUActivationLayer());
        layers.push_back(new MaxPoolingLayer(2, 2));

        layers.push_back(new ConvolutionalLayer(generateRandomFilters(256, 128, 3, 3), 1, 1));
        layers.push_back(new ReLUActivationLayer());
        layers.push_back(new ConvolutionalLayer(generateRandomFilters(256, 256, 3, 3), 1, 1));
        layers.push_back(new ReLUActivationLayer());
        layers.push_back(new ConvolutionalLayer(generateRandomFilters(256, 256, 3, 3), 1, 1));
        layers.push_back(new ReLUActivationLayer());
        layers.push_back(new ConvolutionalLayer(generateRandomFilters(256, 256, 3, 3), 1, 1));
        layers.push_back(new ReLUActivationLayer());
        layers.push_back(new MaxPoolingLayer(2, 2));

        layers.push_back(new FlattenLayer());

        // Fully connected layers
        layers.push_back(new FullyConnectedLayer(generateRandom2DData(4096, 256*7*7), generateRandom1DData(4096)));
        layers.push_back(new FCReLUActivationLayer());
        layers.push_back(new FullyConnectedLayer(generateRandom2DData(4096, 4096), generateRandom1DData(4096)));
        layers.push_back(new FCReLUActivationLayer());
        layers.push_back(new FullyConnectedLayer(generateRandom2DData(1000, 4096), generateRandom1DData(1000)));
        layers.push_back(new SoftmaxActivationLayer());
    }

    // Forward pass through the network
    std::vector<float> forward(const std::vector<std::vector<std::vector<float>>>& input) {
        std::vector<std::vector<std::vector<float>>> output3D = input;
        std::vector<float> output1D;

        bool isFlattened = false;

        // Iterate over layers
        for (Layer* layer : layers) {
            if (isFlattened || dynamic_cast<FullyConnectedLayer*>(layer) != nullptr || dynamic_cast<FCReLUActivationLayer*>(layer) != nullptr || dynamic_cast<SoftmaxActivationLayer*>(layer) != nullptr) {
                output1D = layer->forward1D(output1D);
                std::cout << "Finished forward pass through layer " << typeid(layer).name() << std::endl;
            } 
            else if (dynamic_cast<ConvolutionalLayer*>(layer) != nullptr || dynamic_cast<ReLUActivationLayer*>(layer) != nullptr || dynamic_cast<MaxPoolingLayer*>(layer) != nullptr) {
                output3D = layer->forward3D(output3D);
            } 
            else if (dynamic_cast<FlattenLayer*>(layer) != nullptr) {
                output1D = layer->forwardFlatten(output3D);
                isFlattened = true;
            }
        }
        return output1D;

    }

private:
    std::vector<Layer*> layers;

    // Convert 3D data to 1D data
    std::vector<float> flatten(const std::vector<std::vector<std::vector<float>>>& input) {
        std::vector<float> output;
        for (const auto& matrix : input) {
            for (const auto& row : matrix) {
                for (const auto& value : row) {
                    output.push_back(value);
                }
            }
        }
        return output;
    }
};


int main() {
    // Instantiate VGG19 model
    ConvNet net;

    // Generate random input data
    // std::vector<std::vector<std::vector<float>>> input = generateRandom3DData(3, 224, 224);
    auto input = loadImageData("/Users/precious/Intelligent-Computing-Systems.git/VGG/Beautiful_demoiselle_(Calopteryx_virgo)_male_3.jpg");
    if (input.empty()) {
        std::cout << "Error: Could not load image" << std::endl;
        return 1;
    }

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Perform forward pass
    std::vector<float> output = net.forward(input);

    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate elapsed time
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Print output
    std::cout << "Output: ";
    for (const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // Print elapsed time
    std::cout << "Time elapsed: " << elapsed.count() << " ms" << std::endl;

    return 0;
}