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
for da, la in train_loader:
    _ = my_inception(Variable(da.cuda()))

trn_inception_features.remove()

val_inception_features = LayerActivations(my_inception.Mixed_7c)
for da, la in val_loader:
    - = my_inception(Variable(da.cuda()))

val_inception_features.remove()


### For DenseNet

trn_densenet_features = []
for d, la in train_loader:
    o = my_densenet(Variable(d.cuda()))
    o = o.view(o.size(0), -1)

    trn_densenet_features.extend(o.cpu().data)

val_densenet_features = []
for d, la in val_loader:
    o = my_densenet(Variable(d.cuda()))
    o = o.view(o.size(0), -1)

    val_densenet_features.extend(o.cpu().data)