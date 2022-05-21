from tkinter import Variable
import torch
import torch.nn as nn
from models import my_resnet, my_inception, my_densenet

### For ResNet

trn_labels = []
trn_resnet_features = []
for d, la in train_loader:
    o = my_resnet(Variable(d.cuda()))
    o = o.view(o.size(0), -1)
    trn_labels.extend(la)
    trn_resnet_features.extend(o.cpu().data)
val_labels = []
val_resnet_features = []
for d, la in val_loader:
    o = my_resnet(Variable(d.cuda()))
    o = o.view(o.size(0), -1)
    val_labels.extend(la)
    val_resnet_features.extend(o.cpu().data)

### For Inception

trn_inception_features = LayerActivations(my_inception.Mixed_7c)

### For DenseNet
