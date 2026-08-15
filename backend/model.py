import torch.nn as nn
class Spam(nn.Module):
    def __init__(self,input_dim):
        super(Spam,self).__init__()
        self.linear1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        return self.sigmoid(self.linear2(x))
     