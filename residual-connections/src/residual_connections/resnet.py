import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        append=True
    ):
        super().__init__()

        # Only first layer takes in conv; downscaling if necessary
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Retain feature map dimensions (H, W).
        # Padding should be 1 by the previous formula
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3,
            padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.append = append

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # Following the spirit of the ResNet paper, option A
        if self.stride != 1 or self.in_channels != self.out_channels:
            identity = identity[:, :, ::self.stride, ::self.stride]

            channel_diff = self.out_channels - self.in_channels

            if self.append:
                identity = F.pad(
                    identity,
                    (0, 0, 0, 0, 0, channel_diff)
                )
            else:
                left = channel_diff // 2
                right = channel_diff - left

                identity = F.pad(
                    identity,
                    (0, 0, 0, 0, left, right)
                )

        return F.relu(out + identity)


# Default: ResNet with 3 residual blocks per stage
# This means 20 layers
class ResNet(nn.Module):
    def __init__(self, n=3):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False) # output shape: (N, 16, 32, 32)
        self.bn1 = nn.BatchNorm2d(16)

        # 32x32 feature maps w/ 16 output channels (filter)
        self.stage1 = nn.Sequential(
            *[ResidualBlock(16, 16) for _ in range(n)]
        )

        # 16x16 feature maps w/ 32 output channels (filters)
        self.stage2 = nn.Sequential(
            ResidualBlock(16, 32, stride=2),
            *[ResidualBlock(32, 32) for _ in range(n - 1)]
        )

        # 8x8 feature maps with 64 output channels (FILTERS)
        self.stage3 = nn.Sequential(
            ResidualBlock(32, 64, stride=2),
            *[ResidualBlock(64, 64) for _ in range(n - 1)]
        )


        self.pool = nn.AvgPool2d(8) # output shape (N, 64, 1, 1)
                                    # AdaptiveAvgPool2d would work too
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(start_dim=1)
        return self.fc(x)
