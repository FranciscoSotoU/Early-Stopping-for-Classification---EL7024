import torch.nn as nn
class MLP(nn.Module):
    def __init__(self, input_dim = 2, hidden_dim = 3, output_dim =1 ):
        super().__init__()
        self.input_layer = nn.Linear(input_dim,hidden_dim)
        self.hidden_layer = nn.Linear(hidden_dim,output_dim)
        self.act_fn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.input_layer(x)
        x = self.act_fn(x)
        x = self.hidden_layer(x)
        x = self.sigmoid(x)
        return x