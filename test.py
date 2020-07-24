import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from util.Config import Config
from util.Data import csv_to_tensor, ImgDataset
from model import CH_net


model = CH_net().cuda()
model.load_state_dict(torch.load('res_net.pth'))
config = Config(model)
config.optimizer = torch.optim.Adam(config.model.parameters(), lr=0.001)
config.lr_optim = torch.optim.lr_scheduler.CosineAnnealingLR(config.optimizer,T_max=32)

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])
test_x = csv_to_tensor('data/test.csv', mode='test')
test_set = ImgDataset(test_x, transform =test_transform)
test_loader = DataLoader(test_set, batch_size = config.batch_size, shuffle=False)

#testing
model.eval()
prediction = []
with torch.no_grad():
  for i,data in enumerate(test_loader):
    test_pred = config.model(data.cuda())
    test_label = np.argmax(test_pred.cpu().data.numpy(),axis=1)
    for y in test_label:
      prediction.append(y)

#write to csv
with open('prediction_0724.csv', 'w') as f:
  f.write('ImageId,Label\n')
  for i,y in enumerate(prediction):
    f.write('{},{}\n'.format(i+1, y))
