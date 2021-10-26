import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self,W,H,C,encoding_length):
        super(SelfAttention, self).__init__()
        self.weightKey = nn.Linear(W*H*C,encoding_length)
        self.weightQuery = nn.Linear(W*H*C,encoding_length)
        self.weightValue = nn.Linear(W*H*C,encoding_length)
        self.encoding_length = encoding_length   


    def forward(self, x):
        x = self.flattenCNNOut(x)
        key = self.weightKey(x)
        query = self.weightQuery(x)
        value = self.weightValue(x)
        key = torch.einsum('ijk->ikj', [key])
        key_query = torch.einsum('ijk,ikj->ijk', [query, key])
        key_query = key_query/np.sqrt(key_query.shape[-1])
        score = F.softmax(key_query,dim=0)
        score = score*value
        return key,score
    
    def flattenCNNOut(self,x):
        # x shape BatchSize X SequenceLenght X Widht X Height X Channel
        # output shape BatchSize X SequenceLenght X WxHxC
        input_shape = list(x.size())
        x =  torch.reshape(
            x, (input_shape[0],input_shape[1],input_shape[2]*input_shape[3]*input_shape[4])
            )
        return x

# if __name__ == "__main__":
#     selfAttension = SelfAttention(2,2,1,5)
#     inp = np.double(np.random.rand(6,10,2,2,1))
#     torch_tensor = torch.tensor(inp).float()
#     selfAttension.forward(torch_tensor)