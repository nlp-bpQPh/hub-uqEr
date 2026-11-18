import torch 
import torch.nn as nn
class TorchModel(nn.Module):
    def __init__(self,vocab_size, input_size,hidden_size,dropout,num_classes):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=input_size)
        self.rnn = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=2,batch_first=True,dropout=dropout)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(in_features=hidden_size,out_features=num_classes)
        self.loss = nn.functional.cross_entropy

    def forward(self,x,y = None):
        mask = (x != 0).unsqueeze(-1)
        x = self.embedding(x)
        h,_ = self.rnn(x)
        h = h.masked_fill(~mask,-1e10)
        pooled = h.max(dim=1)[0]
        normed = self.norm(pooled)
        dropout = self.dropout(normed)
        logits = self.linear(dropout)
        if y is not None:
            loss = self.loss(logits,y)
            return loss
        return logits