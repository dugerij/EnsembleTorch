from tkinter import Variable
import torch
import torch.nn as nn

from data_laoder import train_loader, val_loader
from models import my_resnet, my_inception, my_densenet

class LayerActivations():
    features=[]

    def __init__(self, model) -> None:
        self.features = []
        self.hook = model.register_forwward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features.extend(output.view(output.size(0), -1).cpu().data)

    def remove(self):
        self.hook.remove()

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
    _ = my_inception(Variable(da.cuda()))

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