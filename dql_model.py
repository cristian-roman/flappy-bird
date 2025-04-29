import torch
import torch.nn as nn

class DQLModel(nn.Module):
    def __init__(self, action_space=2, input_shape=(1, 72, 128)):
        super(DQLModel, self).__init__()
        self.input_shape = input_shape
        self.action_space = action_space

        # Convolutional layers with BatchNorm
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2)
        self.bn1 = nn.BatchNorm2d(4)

        self.conv2 = nn.Conv2d(4, 8, kernel_size=2)
        self.bn2 = nn.BatchNorm2d(8)

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(8, 16, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(16)

        conv_out_size = self._get_conv_out(self.input_shape)

        self.passage = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
            nn.ReLU()
        )

        self.output_layer = nn.Linear(8, action_space)

        # Initialize weights
        self._init_weights()

    def _get_conv_out(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.max_pool(x)
            x = self.conv3(x)
            x = self.max_pool(x)
            return int(torch.flatten(x, 1).size(1))

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = self.max_pool(x)
        x = self.bn3(self.conv3(x))
        x = self.max_pool(x)
        x = torch.flatten(x, 1)
        x = self.passage(x)
        x = self.output_layer(x)
        return x
