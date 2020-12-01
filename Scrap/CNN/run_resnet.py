import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

from tqdm import tqdm

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torchvision.models as models
from torchvision import datasets, transforms as T

resnet50 = models.resnet50(pretrained=True)

sst_normed = np.load('../../CESM_data/CESM_SST_normalized_lat_weighted.npy').astype(np.float32)
sss_normed = np.load('../../CESM_data/CESM_SSS_normalized_lat_weighted.npy').astype(np.float32)

lead = 12

tstep = 1032
max_epochs = 2
batch_size = 5
channels = 2


y = np.mean(sst_normed[:,lead:,:,:],axis=(2,3)).reshape((tstep-lead)*42,1)  
X = np.transpose(
    np.array([sst_normed,sss_normed])[:,:,0:tstep-lead,:,:].reshape(channels,(tstep-lead)*42,33,89),
    (1,0,2,3))


X_0pad = np.zeros(((tstep-lead)*42,channels,224,224))
X_0pad[:,:,112-17:112+16,112-45:112+44] = X

percent_train = 0.8

X_train = torch.from_numpy( X_0pad[0:int(np.floor(percent_train*(tstep-lead)*1)),:,:,:].astype(np.float32) )
y_train = torch.from_numpy( y[0:int(np.floor(percent_train*(tstep-lead)*1)),:].astype(np.float32) )

X_val = torch.from_numpy( X_0pad[int(np.floor(percent_train*(tstep-lead)*1)):,:,:,:].astype(np.float32) )
y_val = torch.from_numpy( y[int(np.floor(percent_train*(tstep-lead)*1)):,:].astype(np.float32) )

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle = True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle = True)


model = nn.Sequential(nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=(1,1)),
                      resnet50,
                      nn.Linear(in_features=1000,out_features=1))
opt = torch.optim.Adam(model.parameters())
#opt = torch.optim.Adadelta(model.parameters())
loss_fn = nn.MSELoss()


epo_train_losses = [loss_fn(model(X_train),y_train)]
epo_val_losses = [loss_fn(model(X_val),y_val)]

for iepoch in tqdm(range(max_epochs)):
    
    batch_train_losses = []
    model.train()
    for x_batch, y_batch in tqdm(train_loader):
        
        y_pred = model(x_batch)
        loss = loss_fn(y_pred.squeeze(), y_batch.squeeze())
        batch_train_losses.append(loss.item())
        loss.backward()
        opt.step()
        opt.zero_grad()
    epo_train_losses.append( sum(batch_train_losses)/len(batch_train_losses) )

    batch_val_losses = []
    with torch.set_grad_enabled(False):
        for x_batch_val, y_batch_val in tdqm(val_loader):
            y_pred = model(x_batch_val)
            loss = loss_fn(y_pred.squeeze(), y_batch_val.squeeze())
            batch_val_losses.append(loss.item())
        epo_val_losses.append( sum(batch_val_losses)/len(batch_val_losses) )


model.eval()

plt.figure(figsize=(6,4))
plt.rcParams.update({'font.size': 15})

plt.plot(epo_train_losses)
plt.plot(epo_val_losses)
plt.legend(['Train loss','Validation loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.tight_layout()

plt.savefig('res_net_loss_at_epoch.pdf')

