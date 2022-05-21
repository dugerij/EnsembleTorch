import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleModel(nn.Module):

    def __init__(self, out_size, training=True):
        super(EnsembleModel, self).__init__()
        self.fc1 = nn.Linear(8192, 512)
        self.fc2 = nn.Linear(131072, 512)
        self.fc3 = nn.Linear(82944, 512)
        self.fc4 = nn.Linear(512, out_size)


    def forward(self, inp1, inp2, inp3):
        out1 = self.fc1(F.dropout(inp1, training=self.training))
        out2 = self.fc2(F.dropout(inp2, training=self.training))
        out3 = self.fc3(F.dropout(inp3, training=self.training))
        out = out1 + out2 + out3
        out = self.fc4(F.dropout(out, training=self.training))
        return out

emodel = EnsembleModel(2)
if torch.Tensor.is_cuda:
    emodel = emodel.cuda()