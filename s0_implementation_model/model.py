import torch
from torch import nn
from torch.nn.functional import relu, log_softmax, max_pool2d


class MyAwesomeModel(nn.Module):
    """"My awesome model"""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = relu(self.conv1(x))
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = relu(self.conv2(x))
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = relu(self.conv3(x))
        x = max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        return x


if __name__ == "__main__":
    model = MyAwesomeModel()
    print(f"Model architecture: {model}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
