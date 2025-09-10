
'''
https://discuss.pytorch.org/t/splitting-up-sequential-batches-into-randomly-shuffled-train-test-subsets/106466/2
Thank you ptrblck for the example code.
'''

from time import time
from random import random, shuffle, seed
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, feature, labels):
        self.features = feature
        self.labels   = labels
        self.len      = len(self.features)
        seed(int(time()))

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
    
    def __len__(self):
        return self.len
