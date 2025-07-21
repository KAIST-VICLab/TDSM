from torch.utils.data import Dataset
import numpy as np


class FeatureFeeder(Dataset):
    def __init__(self, path, split='train'):
        if split == 'train':
            self.x = np.load(path + '/train.npy')
            self.y = np.load(path + '/train_label.npy')
        elif split == 'val':
            self.x = np.load(path + '/ztest.npy')
            self.y = np.load(path + '/z_label.npy')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], int(self.y[index])