import torch
import torch.nn as nn
import torchvision
from torchvision import models

# create ResNet model

my_resnet = models.resnet34(pretrained=True)

if torch.Tensor.is_cuda:
     my_resnet = my_resnet.cuda()

my_resnet = nn.Sequential(*list(my_resnet.children()))[-1]

for p in my_resnet.parameters():
    p.requires_grad = False

# Create inception model

my_inception = models.inception_v3(pretrained=True)
my_inception.aux_logits = False

if torch.Tensor.is_cuda:
    my_inception = my_inception.cuda()
for p in my_inception.parameters():
    p.requires_grad = False


# Create DenseNet model
my_densenet = models.densenet121(pretrained=True).features

if torch.Tensor.is_cuda:
    my_densenet = my_densenet.cuda()

for p in my_densenet.parameters():
    p.requires_grad = False

