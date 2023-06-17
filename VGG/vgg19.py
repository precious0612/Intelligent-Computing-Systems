import numpy as np
import scipy.io
import cv2

import time

class VGG19:
    def __init__(self, input_shape=(224, 224, 3), num_classes=1000):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.layers = []
        
    def add_flatten_layer(self):
        flatten_layer = {
            'type': 'flatten'
        }
        self.layers.append(flatten_layer)

    def add_conv_layer(self, filters, kernel_size, activation='relu', padding='same'):
        conv_layer = {
            'type': 'conv',
            'filters': filters,
            'kernel_size': kernel_size,
            'activation': activation,
            'padding': padding
        }
        self.layers.append(conv_layer)

    def add_maxpool_layer(self, pool_size, strides=None):
        maxpool_layer = {
            'type': 'maxpool',
            'pool_size': pool_size,
            'strides': strides
        }
        self.layers.append(maxpool_layer)

    def add_dense_layer(self, units, activation='relu'):
        dense_layer = {
            'type': 'dense',
            'units': units,
            'activation': activation
        }
        self.layers.append(dense_layer)

    def add_output_layer(self):
        output_layer = {
            'type': 'output'
        }
        self.layers.append(output_layer)

    def build(self):
        model = []
        prev_channels = self.input_shape[-1]

        for layer in self.layers:
            layer_type = layer['type']

            if layer_type == 'conv':
                filters = layer['filters']
                kernel_size = layer['kernel_size']
                activation = layer['activation']
                padding = layer['padding']
                weights = np.random.randn(kernel_size[0], kernel_size[1], prev_channels, filters) * 0.01
                bias = np.zeros((1, 1, 1, filters))

                conv_layer = {
                    'type': 'conv',
                    'weights': weights,
                    'bias': bias,
                    'activation': activation,
                    'padding': padding
                }
                model.append(conv_layer)
                prev_channels = filters

            elif layer_type == 'maxpool':
                pool_size = layer['pool_size']
                strides = layer['strides'] or pool_size

                maxpool_layer = {
                    'type': 'maxpool',
                    'pool_size': pool_size,
                    'strides': strides
                }
                model.append(maxpool_layer)

            elif layer_type == 'dense':
                units = layer['units']
                activation = layer['activation']
                weights = np.random.randn(prev_channels, units) * 0.01
                bias = np.zeros((1, units))

                dense_layer = {
                    'type': 'dense',
                    'weights': weights,
                    'bias': bias,
                    'activation': activation
                }
                model.append(dense_layer)
                prev_channels = units
                
            elif layer_type == 'flatten':
                flatten_layer = {
                    'type': 'flatten'
                }
                model.append(flatten_layer)
                prev_channels = prev_channels * 7 * 7

            elif layer_type == 'output':
                output_layer = {
                    'type': 'output'
                }
                model.append(output_layer)

        self.model = model
        
    def load_weights(self, weights):
        for i, layer in enumerate(self.layers):
            if layer['type'] == 'conv' or layer['type'] == 'dense':
                layer_weights = weights[i][0][0][0][0][0]
                layer_bias = weights[i][0][0][0][0][1]
                self.layers[i]['weights'] = layer_weights
                self.layers[i]['bias'] = layer_bias
                
    def flatten(self, x):
        return x.reshape(-1)

    def padding(self, array, pad_width, mode, **kwargs):
        if pad_width == 0:
            return array
        if mode == 'constant':
            array = np.pad(array, pad_width, mode, **kwargs)
        return array

    def calculate_lenth(self, array_lenth, kernel_lenth, stride):
        return int((array_lenth - kernel_lenth) / stride + 1)

    def conv2d(self, x, weights, bias, padding='same'):
        kernel_size = weights.shape[:2]
        if len(x.shape) == 4:
            x = x[0]
            
        if padding == 'same':
            pad = (kernel_size[0] - 1) // 2
            channel = x.shape[0]
            result = []
                
            for i in range(channel):
                result.append(self.padding(x[i], pad, 'constant', constant_values=0))
                
            x = np.array(result)
        elif padding == 'valid':
            pass  # No padding needed for 'valid' convolution
        output_channel = weights.shape[-1]
        kernel = weights.reshape(-1, output_channel)
        width = self.calculate_lenth(x.shape[1], kernel_size[-2], 1)
        height = self.calculate_lenth(x.shape[2], kernel_size[-1], 1)
        output = np.zeros(((output_channel, width, height)))
        for i in range(width):
            for j in range(height):
                array_comp = x[:,i:i+kernel_size[-2], j:j+kernel_size[-1]].flatten()
                output[:, i, j] = np.dot(array_comp, kernel.flatten().reshape(output_channel, -1).T) + bias
                
        return output

    def maxpool2d(self, x, pool_size, strides):
        channels, height, width = x.shape
        p_height, p_width = pool_size
        s_height, s_width = strides
        out_height = int((height - p_height) / s_height) + 1
        out_width = int((width - p_width) / s_width) + 1
        out = np.zeros((channels, out_height, out_width))

        for h in range(out_height):
            for w in range(out_width):
                for c in range(channels):
                    receptive_field = x[c, h * s_height : h * s_height + p_height, w * s_width : w * s_width + p_width]
                    out[c, h, w] = np.max(receptive_field)

        return out

    def dense(self, x, weights, bias, activation='relu'):
        x = x.reshape(-1)
        out = np.dot(weights.T, x) + bias
        if activation == 'relu':
            out[out < 0] = 0
        elif activation == 'tanh':
            out = np.tanh(out)
        elif activation == 'softmax':
            out = self.softmax(out)

        return out

    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    def forward(self, x):
        for layer in self.model:
            layer_type = layer['type']
            print("calculating layer: ", layer_type, " ...")
            start_time = time.time()

            if layer_type == 'conv':
                weights = layer['weights']
                bias = layer['bias']
                activation = layer['activation']
                padding = layer['padding']
                x = self.conv2d(x, weights, bias, padding)
                x = self.relu(x) if activation == 'relu' else self.tanh(x)

            elif layer_type == 'maxpool':
                pool_size = layer['pool_size']
                strides = layer['strides']
                x = self.maxpool2d(x, pool_size, strides)

            elif layer_type == 'dense':
                weights = layer['weights']
                bias = layer['bias']
                activation = layer['activation']
                x = self.dense(x, weights, bias, activation)
                
            elif layer_type == 'flatten':
                x = self.flatten(x)

            elif layer_type == 'output':
                x = self.softmax(x)
                
            end_time = time.time()
            print(layer_type, " time: ", end_time - start_time)

        return x

# Load ImageNet class labels
class_labels = scipy.io.loadmat('/Users/precious/Intelligent-Computing-Systems.git/VGG/ILSVRC2012_devkit_t12/data/meta.mat')
class_labels = [c[0][2][0] for c in class_labels['synsets']]
# print(class_labels)

# Load pre-trained weights of VGG19
weights = scipy.io.loadmat('imagenet-vgg-verydeep-19.mat')
weights = weights['layers'][0]

# Create VGG19 model
vgg19 = VGG19()
vgg19.add_conv_layer(64, (3, 3), activation='relu', padding='same')
vgg19.add_conv_layer(64, (3, 3), activation='relu', padding='same')
vgg19.add_maxpool_layer((2, 2))
vgg19.add_conv_layer(128, (3, 3), activation='relu', padding='same')
vgg19.add_conv_layer(128, (3, 3), activation='relu', padding='same')
vgg19.add_maxpool_layer((2, 2))
vgg19.add_conv_layer(256, (3, 3), activation='relu', padding='same')
vgg19.add_conv_layer(256, (3, 3), activation='relu', padding='same')
vgg19.add_conv_layer(256, (3, 3), activation='relu', padding='same')
vgg19.add_conv_layer(256, (3, 3), activation='relu', padding='same')
vgg19.add_maxpool_layer((2, 2))
vgg19.add_conv_layer(512, (3, 3), activation='relu', padding='same')
vgg19.add_conv_layer(512, (3, 3), activation='relu', padding='same')
vgg19.add_conv_layer(512, (3, 3), activation='relu', padding='same')
vgg19.add_conv_layer(512, (3, 3), activation='relu', padding='same')
vgg19.add_maxpool_layer((2, 2))
vgg19.add_conv_layer(512, (3, 3), activation='relu', padding='same')
vgg19.add_conv_layer(512, (3, 3), activation='relu', padding='same')
vgg19.add_conv_layer(512, (3, 3), activation='relu', padding='same')
vgg19.add_conv_layer(512, (3, 3), activation='relu', padding='same')
vgg19.add_maxpool_layer((2, 2))
vgg19.add_flatten_layer()
vgg19.add_dense_layer(4096, activation='relu')
vgg19.add_dense_layer(4096, activation='relu')
vgg19.add_dense_layer(vgg19.num_classes, activation='softmax')

# Build VGG19 model
vgg19.build()
vgg19.load_weights(weights)

# # randomly generate image
# image = np.random.randint(0, 1, (3, 224, 224))

# process image
def process_image(img):
    img = cv2.resize(img, (224, 224))
    return img

image = cv2.imread('/Users/precious/Intelligent-Computing-Systems.git/VGG/Beautiful_demoiselle_(Calopteryx_virgo)_male_3.jpg')
image = process_image(image)

start_time = time.time()
# Forward pass through the VGG19 network
output = vgg19.forward(image)
end_time = time.time()

print("Total time taken: ", end_time - start_time)

# Get the top-5 predicted class labels and probabilities
top5_indices = np.argsort(output)[0, -5:][::-1]
top5_labels = [class_labels[i] for i in top5_indices]
top5_probabilities = output[0][top5_indices]

# Print the top-5 predicted class labels and probabilities
for label, prob in zip(top5_labels, top5_probabilities):
    print(f'{label}: {prob}')
