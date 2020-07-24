from torch.utils.data import Dataset
import pandas as pd
import torch

#csv to feature(tensor) and label(numpy)
def csv_to_tensor(path, mode='train'):
  data = pd.read_csv(path).values
  if mode=='train':
    features = torch.from_numpy(data[:,1:].reshape(-1,28,28)).type(torch.float)
    label = torch.LongTensor(data[:,0])
    return features, label
  elif mode=='test':
    return torch.from_numpy(data.reshape(-1,28,28)).type(torch.float)

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        self.y = y
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

if __name__ =='__main__':
    import torchvision.transforms as transforms
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])
    data_features,data_label = csv_to_tensor('../test_case/test_case.csv', mode='train')
    train_set = ImgDataset(data_features, data_label, transform = train_transform)
    print('done')
