import torch.nn as nn

class CH_net(nn.Module):
  def __init__(self):
    super(CH_net,self).__init__()
    self.model1 = nn.Sequential(
        nn.Conv2d(1,64,4,2,3),    #28->16
        nn.LeakyReLU(),
        nn.BatchNorm2d(64),
    )
    self.model2 = nn.Sequential(
        nn.Conv2d(64,128,4,2,1),  #16->8
        nn.LeakyReLU(),
        nn.BatchNorm2d(128),

        nn.Conv2d(128,128,3,1,1),    #8->8
        nn.LeakyReLU(),
        nn.BatchNorm2d(128),
    )
    self.model3 = nn.Sequential(
        nn.Conv2d(128,256,4,2,1),    #8->4
        nn.LeakyReLU(),
        nn.BatchNorm2d(256),

        nn.Conv2d(256,256,3,1,1),    #4->4
        nn.LeakyReLU(),
        nn.BatchNorm2d(256),
    )
    self.model4 = nn.Sequential(
        nn.Conv2d(256,512,4,2,1),   #4->2
        nn.LeakyReLU(),
        nn.BatchNorm2d(512),

        nn.Conv2d(512,10,2,1,0),      #2->1
    )
    self.softmax = nn.Softmax(dim=1)
    self.downsampling2 = nn.Sequential(
        nn.Conv2d(64,128,1,2,0),
        nn.BatchNorm2d(128),
    )
    self.downsampling3 = nn.Sequential(
        nn.Conv2d(128,256,1,2,0),
        nn.BatchNorm2d(256),
    )
    
  def forward(self, input):
    buffer1 = self.model1(input)
    buffer2 = self.model2(buffer1)+self.downsampling2(buffer1)
    buffer3 = self.model3(buffer2)+self.downsampling3(buffer2)
    output = self.model4(buffer3).squeeze(3).squeeze(2)    #(batch_size,10,1,1)
    output = self.softmax(output)
    return output
