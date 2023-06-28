import functools
import operator

import torch

def get_output_shape(module, input_shape):
    return module(torch.rand(*input_shape)).shape

def get_output_dim(module, input_shape):
    return functools.reduce(operator.mul, get_output_shape(module, input_shape))

class FcResidualBlock(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.fc1 = torch.nn.Linear(dim, dim)
        self.bn1 = torch.nn.BatchNorm1d(dim)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(dim, dim)
        self.bn2 = torch.nn.BatchNorm1d(dim)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + x
        out = self.relu2(out)
        return out

class ConvResidualBlock(torch.nn.Module):
    def __init__(self, channel_num, kernel_size):
        super().__init__()

        assert kernel_size % 2 == 1
        self.conv1 = torch.nn.Conv2d(channel_num, channel_num, kernel_size, padding=kernel_size // 2)
        self.bn1 = torch.nn.BatchNorm2d(channel_num)
        self.relu1 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(channel_num, channel_num, kernel_size, padding=kernel_size // 2)
        self.bn2 = torch.nn.BatchNorm2d(channel_num)
        self.relu2 = torch.nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.relu2(out)
        return out

class QNetwork(torch.nn.Module):
    def __init__(self, state_shape, action_dim, backbone):
        super().__init__()

        dim = get_output_dim(backbone, (2, *state_shape[1:])) // 2
        self.layers = torch.nn.Sequential(
            backbone,
            torch.nn.Linear(dim, action_dim),
            )

    def forward(self, x):
        return self.layers(x)

class PNetwork(torch.nn.Module):
    def __init__(self, state_shape, action_dim, backbone):
        super().__init__()

        dim = get_output_dim(backbone, (2, *state_shape[1:])) // 2
        self.layers = torch.nn.Sequential(
            backbone,
            torch.nn.Linear(dim, action_dim),
            )

    def forward(self, x):
        return self.layers(x)

class PVNetwork(torch.nn.Module):
    def __init__(self, state_shape, action_dim, backbone):
        super().__init__()

        self.backbone = backbone
        dim1 = get_output_dim(self.backbone, (2, *state_shape[1:])) // 2
        dim2 = ((int((state_shape[2] * state_shape[3] * action_dim) ** 0.5) + 15) // 16) * 16
        self.vhead = torch.nn.Sequential(
            torch.nn.Linear(dim1, dim2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim2, 1),
            )
        self.phead = torch.nn.Sequential(
            torch.nn.Linear(dim1, dim2),
            torch.nn.ReLU(),
            torch.nn.Linear(dim2, action_dim),
            )

    def forward(self, x):
        x = self.backbone(x)
        return self.phead(x), self.vhead(x)
