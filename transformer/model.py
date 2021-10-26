import torch
import torch.nn as nn
import torch.nn.functional as F
from attention.attention import SelfAttention
import numpy as np

class Model(nn.Module):
    def __init__(self,W,H,C,encoding_length):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)
        self.attention1 = SelfAttention(W,H,C,encoding_length)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        print(x.shape,"***************")
        keyEnc,valueEnc = self.attention1(x)
        print(keyEnc.shape)
        return keyEnc

if __name__ == "__main__":
    model = Model(2,2,1,5)
    inp = np.double(np.random.rand(6,1,10,10))
    torch_tensor = torch.tensor(inp).float()
    model.forward(torch_tensor)