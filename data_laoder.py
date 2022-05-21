from tkinter import Variable
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn


from data_laoder import trn_densenet_features, trn_inception_features, trn_resnet_features, trn_labels
from data_laoder import val_resnet_features, val_inception_features, val_densenet_features, val_labels

from models import my_resnet, my_inception, my_densenet

data_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for dogs and cat dataset
train_dset = ImageFolder('dogsandcats/train/', transform=data_transform)
val_dset = ImageFolder('dogsandcats/valid/', transform=data_transform)

train_loader = DataLoader(train_dset, batch_size=32, shuffle=False, num_workers=3)
val_loader = DataLoader(val_dset, batch_size=32, shuffle=False, num_workers=3)


class FeaturesDataset(Dataset):
    def __init__(self, featlst1, featlst2, featlst3, labellst):
        super(FeaturesDataset, self).__init__()
        self.featlst1 = featlst1
        self.featlst2 = featlst2
        self.featlst3 = featlst3
        self.labellst = labellst

    def __getitem__(self, index):
        return self.featlst1[index], self.featlst2[index], self.featlst3[index], self.labellst[index]

    def __len__(self):
        return len(self.labellst)

trn_feat_dset = FeaturesDataset(trn_resnet_features, trn_inception_features.features, trn_densenet_features, trn_labels)
val_feat_dset = FeaturesDataset(val_resnet_features, val_inception_features.features, val_densenet_features, val_labels)

trn_feat_loader = DataLoader(trn_feat_dset, batch_size=64, shuffle=True)
val_feat_loader = DataLoader(val_feat_dset, batch_size=64)

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