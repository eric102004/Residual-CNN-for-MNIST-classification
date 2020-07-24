import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import time

from util.Config import Config
from util.Data import csv_to_tensor, ImgDataset
from model import CH_net

train_x, train_y = csv_to_tensor('data/train.csv', mode='train')


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.RandomAffine(10, translate=(0.1,0.1)),
    transforms.RandomRotation(15),
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
])

model = CH_net().cuda()
config = Config(model)
config.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.001)
config.lr_optim = torch.optim.lr_scheduler.CosineAnnealingLR(config.optimizer,T_max=32)

train_set = ImgDataset(train_x, train_y, transform = train_transform)
train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)

#start training
config.epoch = 30
change1=0
change2=0
for epoch in range(config.epoch):
  '''
  if not change1 and epoch>=config.epoch*1/3:
      config.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.0003)
      change1=1
  if not change2 and epoch>=config.epoch*2/3:
      config.optimizer = torch.optim.SGD(config.model.parameters(), lr=0.00005)
      change2=1
  '''
  epoch_start_time = time.time()
  train_loss = 0.0
  train_acc  = 0.0
  config.model.train()
  for i,data in enumerate(train_loader):
    config.optimizer.zero_grad()
    train_pred = config.model(data[0].cuda())
    batch_loss = config.loss(train_pred, data[1].cuda())
    batch_loss.backward()
    config.optimizer.step()

    train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1)==data[1].numpy())
    train_loss += batch_loss.item()

  config.lr_optim.step()
   
  print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f' % \
      (epoch + 1, config.epoch, time.time()-epoch_start_time, \
       train_acc/train_set.__len__(), train_loss/train_set.__len__()))

#save model
torch.save(config.model.state_dict(),'res_net.pth')

