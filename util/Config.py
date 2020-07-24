class Config:
  def __init__(self, model):
    self.batch_size = 128
    self.epoch = 30
    self.model = model
    self.loss = nn.CrossEntropyLoss()
    self.optimizer = None
    self.lr_optim = None
    self.work_dic = './data'
