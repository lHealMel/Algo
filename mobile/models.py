import torch.nn as nn

class MNet(nn.Module):
    def __init__(self):
        super(MNet, self).__init__()
        self.il = nn.Linear(8, 128)
        self.relu = nn.ReLU()
        # self.hL = nn.Linear(128, 64)
        # self.relu = nn.ReLU()
        self.oL = nn.Linear(128, 8)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.il(x)
        x = self.relu(x)
        # x = self.hL(x)
        # x = self.relu(x)
        x = self.oL(x)
        x = self.relu(x)
        return x