import numpy as np

def padding(array, pad_width, mode, **kwargs):
    if pad_width == 0:
        return array
    if mode == 'constant':
        array = np.pad(array, pad_width, mode, **kwargs)
    return array

def calculate_lenth(array_lenth, kernel_lenth, stride):
    return int((array_lenth - kernel_lenth) / stride + 1)

def convolve(array, kernel, bias, pad:int, stride:int = 1, **kwargs):
    if pad:
        array = padding(array, pad, 'constant', constant_values=0)
        
    kernel_size = kernel.shape
    width = calculate_lenth(array.shape[0], kernel_size[0], stride)
    height = calculate_lenth(array.shape[1], kernel_size[1], stride)
    result = np.zeros(((width, height)))
    
    for i in range(width):
        x = i * stride
        for j in range(height):
            y = j * stride
            if i + kernel_size[0] <= len(array) and j + kernel_size[1] <= len(array[0]):
                result[i, j] = np.dot(array[x:x+kernel_size[0], y:y+kernel_size[1]].flatten(), kernel.flatten()) + bias
    return result

array = np.random.rand(3,4)
print("原始矩阵：", array)
bias = np.ones(1)
print("bias：", bias)
result = convolve(array, np.ones((2,2)), bias, pad=1, stride=2)
print("卷积操作：", result)


def channels_convolve(array, kernel, bias, pad:int = 0, stride:int = 1, channel_last=False, **kwargs):
    if channel_last:
        array = array.transpose(1, 2, 0)
        
    channel = len(array)
    result = []
        
    for i in range(channel):
        result.append(padding(array[i], pad, 'constant', constant_values=0))
        
    array = np.array(result)
        
    kernel_size = kernel.shape
    output_channel = kernel_size[0]
    width = calculate_lenth(array.shape[1], kernel_size[-2], stride)
    height = calculate_lenth(array.shape[2], kernel_size[-1], stride)
    result = np.zeros(((output_channel, width, height)))
    
    for i in range(width):
        x = i * stride
        for j in range(height):
            y = j * stride
            if i + kernel_size[-2] <= len(array[0]) and j + kernel_size[-1] <= len(array[1]):
                array_comp = array[:, x:x+kernel_size[-2], y:y+kernel_size[-1]].flatten()
                result[:, i, j] = np.dot(array_comp, kernel.flatten().reshape(output_channel, -1).T) + bias
    return result

array = np.random.rand(3,3,4)
print("原始矩阵：", array)
kernel = np.ones((3,3,2,2))
print("kernel：", kernel)
bias = np.ones(3)
print("bias：", bias)

result = channels_convolve(array, kernel, bias, pad=1, stride=2, channel_last=False)
print("多通道卷积：", result)

def batch_channels_convolve(array, kernel, bias, pad=1, stride=2, channel_last=False):
    output = []
    for i in array:
        output.append(channels_convolve(i, kernel, bias, pad, stride, channel_last))
    
    return np.array(output, dtype=np.float32)

batch = 10
print("批次：", batch)
array = np.random.rand(batch,3,3,4)
print("原始矩阵：", array)
kernel = np.ones((3,3,2,2))
print("kernel：", kernel)
bias = np.ones(3)
print("bias：", bias)

result = batch_channels_convolve(array, kernel, bias)
print("批次多通道卷积：", result)
