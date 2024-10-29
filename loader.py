import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

class EEGDataset(Dataset): 
    def __init__(self, data_file, transform=None):
        self.data = pd.read_csv(data_file)
        
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = self.data.iloc[idx, 1:].values.astype(np.float32).reshape(4, 440)
        y = self.data.iloc[idx, 0]

        if y == -1:
            y = 10
        
        if self.transform:
            x = self.transform(x)
        # convert data in numpy array to tensor
        x = torch.tensor(x)
        y = torch.tensor(y)
        return x, y