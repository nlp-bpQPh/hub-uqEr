import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

class TextDataset(Dataset):
    def __init__(self,vocab,rows,max_len,label2id):
        self.X = [self.encode(row[0],vocab,max_len) for row in rows]
        self.y = [label2id.get(row[1]) for row in rows]
    def __getitem__(self,index):
        return torch.tensor(self.X[index]),torch.tensor(self.y[index])
    def __len__(self):
        return len(self.y)

    def encode(self,sentence,vocab,max_len):
        ids = [vocab.get(char,1) for char in sentence][:max_len]
        ids += [0] * (max_len - len(ids))

        return ids