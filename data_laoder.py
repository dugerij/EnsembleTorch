import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from .data_preprocess import trn_densenet_features, trn_inception_features, trn_resnet_features, trn_labels
from .data_preprocess import val_resnet_features, val_inception_features, val_densenet_features, val_labels

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