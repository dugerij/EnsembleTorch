from tkinter import Variable
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .EnsembleModel import emodel
from .data_laoder import trn_feat_loader, val_feat_loader

optimizer = Adam(emodel.parameters(), lr=0.01)

def fit(epoch, model, data_loader, phase='training', volatile=False):
    if phase == 'training':
        model.train()
    if phase == 'validation':
        model.eval()
        volatile=True
    running_loss = 0.0
    running_correct = 0
    for batch_idx, (data1, data2, data3, target) in enumerate(data_loader):
        if torch.Tensor.is_cuda:
            data1, data2, data3, target = data1.cuda(), data2.cuda(), data3.cuda(), target.cuda()
        data1, data2, data3, target = Variable(data1, volatile), Variable(data2, volatile), Variable(data3, volatile), Variable(target)
        if phase == 'training':
            optimizer.zero_grad()
        output = model(data1, data2, data3)
        loss = F.cross_entropy(output, target)

        running_loss += F.cross_entropy(output, target, size_average=False).data[0]
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum()
        if phase == 'training':
            loss.backward()
            optimizer.step()
    
    loss = running_loss / len(data_loader.dataset)
    accuracy = 100. * running_correct / len(data_loader.dataset)

    print(f'{phase} loss {loss:{5}.{2}} and {phase} accuracy is {running_correct}/{len(data_loader.dataset)}{accuracy:{10}.{4}}')


    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []

    for epoch in range(1, 10):
        epoch_loss, epoch_accuracy = fit(epoch, emodel, trn_feat_loader, phase='training')
        val_epoch_loss, val_epoch_accuracy = fit(epoch, emodel, val_feat_loader, phase='validation')

        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)