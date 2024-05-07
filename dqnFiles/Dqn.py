import torch
from torch import nn


class Dqn(nn.Module):
    def __init__(self, actions=2):
        super(Dqn, self).__init__()
        self.layer1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.relu1 = nn.ReLU(inplace=True)
        self.layer2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.layer3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(512, actions)

    def forward(self, data):
        output = self.layer1(data)
        output = self.layer2(output)
        output = self.layer3(output)

        #flatten
        output = output.view(output.size(0), -1)

        output = self.fc1(output)
        output = self.relu4(output)
        output = self.fc2(output)
        # print(model)

        return output

    def set_weights(self):
        for layer in [self.layer1,  self.layer2, self.layer3]:
            torch.nn.init.normal(layer.weight, mean=0.0, std=0.01)
            if layer.bias is not None:
                layer.bias.data.fill_(0.01)

        for layer in [self.fc1, self.fc2]:
            torch.nn.init.normal(layer.weight, mean=0.0, std=0.01)
            layer.bias.data.fill_(0.01)
