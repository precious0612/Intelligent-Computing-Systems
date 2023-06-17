import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import PackedSequence

# VGG19 forward
import torchvision.models as models

# 下载并加载预训练的 VGG19 模型
vgg19 = models.vgg19(weights='IMAGENET1K_V1')

# process image
import torchvision.transforms as transforms
from PIL import Image

# 读取图片
def load_and_preprocess_image(image_path):
    # Load image using PIL
    image = Image.open(image_path)
    
    # Preprocess image
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    
    # Add batch dimension
    image = image.unsqueeze(0)
    
    return image

# Load and preprocess the test image
image = load_and_preprocess_image('/Users/precious/Intelligent-Computing-Systems.git/VGG/Fox_Squirrel_(14539535789).jpg')  # Replace 'test_image.jpg' with your image path

import time
start = time.time()
features = vgg19.features(image)
end = time.time()

print('VGG19 forward time: ', end - start)

# on GPU

vgg19 = models.vgg19(weights='IMAGENET1K_V1').to('mps')
image = load_and_preprocess_image('/Users/precious/Intelligent-Computing-Systems.git/VGG/Fox_Squirrel_(14539535789).jpg').to('mps')
start = time.time()
features = vgg19.features(image)
end = time.time()

print('VGG19 forward time on GPU: ', end - start)