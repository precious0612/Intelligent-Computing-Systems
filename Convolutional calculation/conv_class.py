import numpy as np

class Convolution:
    """基本的卷积操作。"""
    def __init__(self, kernel, bias, pad:int, stride:int = 1):
        self.kernel = kernel
        self.bias = bias
        self.pad = pad
        self.stride = stride

    def padding(self, array, pad_width, mode='constant', **kwargs):
        """根据pad_width给array增加padding"""
        if pad_width == 0:
            return array
        if mode == 'constant':
            array = np.pad(array, pad_width, mode, **kwargs)
        return array

    def calculate_length(self, array_length, kernel_length, stride, **kwargs):
        """计算卷积后的长度"""
        return int((array_length - kernel_length) / stride + 1)

    def forward(self, array):
        """前向传播"""
        if self.pad:
            array = self.padding(array, self.pad, 'constant', constant_values=0)
            
        kernel_size = self.kernel.shape
        width = self.calculate_length(array.shape[0], kernel_size[0], self.stride)
        height = self.calculate_length(array.shape[1], kernel_size[1], self.stride)
        result = np.zeros(((width, height)))
        
        for i in range(width):
            x = i * self.stride
            for j in range(height):
                y = j * self.stride
                if i + kernel_size[0] <= len(array) and j + kernel_size[1] <= len(array[0]):
                    result[i, j] = np.dot(array[x:x+kernel_size[0], y:y+kernel_size[1]].flatten(), self.kernel.flatten()) + self.bias
        return result

class ChannelsConvolution(Convolution):
    """多通道卷积，继承自基础卷积操作。"""
    def __init__(self, kernel, bias, pad:int, stride:int = 1, channel_last=False):
        super().__init__(kernel, bias, pad, stride)
        self.channel_last = channel_last

    def forward(self, array):
        """前向传播"""
        if self.channel_last:
            array = array.transpose(1, 2, 0)  # Change to (channel, height, width)
        
        channel = len(array)
        temp = []
        for i in range(channel):
            temp.append(self.padding(array[i], self.pad, 'constant', constant_values=0))
        
        array = np.array(temp)
        
        kernel_size = self.kernel.shape
        output_channel = kernel_size[0]
        width = self.calculate_length(array.shape[1], kernel_size[-2], self.stride)
        height = self.calculate_length(array.shape[2], kernel_size[-1], self.stride)
        result = np.zeros((output_channel, width, height))
        
        for i in range(width):
            x = i * self.stride
            for j in range(height):
                y = j * self.stride
                if i + kernel_size[-2] <= len(array[0]) and j + kernel_size[-1] <= len(array[1]):
                    array_comp = array[:, x:x+kernel_size[-2], y:y+kernel_size[-1]].flatten()
                    result[:, i, j] = np.dot(array_comp, self.kernel.flatten().reshape(output_channel, -1).T) + self.bias
        return result


class BatchChannelsConvolution(ChannelsConvolution):
    """批次多通道卷积，继承自多通道卷积操作。"""
    def __init__(self, kernel, bias, pad:int, stride:int = 1, channel_last=False):
        super().__init__(kernel, bias, pad, stride, channel_last)

    def forward(self, batch_array):
        """前向传播"""
        output = []
        for i in batch_array:
            output.append(ChannelsConvolution(self.kernel, self.bias, self.pad, self.stride, self.channel_last).forward(i))
    
        output = np.array(output, dtype=np.float32)
        return output


# 测试代码

# 随机初始化参数
array = np.random.rand(3,4)
kernel = np.ones((2,2))
bias = np.ones(1)

# 创建一个卷积对象
conv = Convolution(kernel, bias, pad=1, stride=2)
result = conv.forward(array)
print("卷积结果：", result)

# 随机初始化参数
array = np.random.rand(3,3,4)
kernel = np.ones((3,3,2,2))
bias = np.ones(3)

# 创建一个多通道卷积对象
multi_conv = ChannelsConvolution(kernel, bias, pad=1, stride=2)
result = multi_conv.forward(array)
print("多通道卷积结果：", result)

# 随机初始化参数
batch = 10    # 批次大小
array = np.random.rand(batch,3,3,4)
kernel = np.ones((3,3,2,2))
bias = np.ones(3)

# 创建一个批次多通道卷积对象
batch_multi_conv = BatchChannelsConvolution(kernel, bias, pad=1, stride=2)
result = batch_multi_conv.forward(array)
print("批次多通道卷积结果：", result)