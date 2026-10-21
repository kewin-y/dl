import torch
import torch.nn as nn
import torch.nn.functional as F

classes = 10

class Subsample2d(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__()

        # Forward the arguments to AvgPool2d
        # Following to the paper, the divisor override of the avg pool should be 1
        # This won't affect training because the weight will be learned anyway
        self.avg_pool = nn.AvgPool2d(*args, **{**kwargs, "divisor_override": 1})

        # Separate weight/bias for each channel
        self.weight = nn.Parameter(torch.randn(in_channels))
        self.bias = nn.Parameter(torch.randn(in_channels))

    def forward(self, x):
        # σ(weight * avg_pool(x) + bias)

        pre = self.avg_pool(x)
        w = self.weight[None, :, None, None]
        b = self.bias[None, :, None, None]
        return torch.sigmoid(w * pre + b)

class C3(nn.Module):
    def __init__(self):
        super().__init__()

        # Sparse connections as seen in the paper.
        # This is done to introduce asymmetry in the hopes that the parameters learn different features
        # Sparse connections also reduce the memory footprint
        self.connections = [
            # Contiguous subsets of 3
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [0, 4, 5],
            [0, 1, 5],

            # Contiguous subsets of 4
            [0, 1, 2, 3],
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [0, 3, 4, 5],
            [0, 1, 4, 5],
            [0, 1, 2, 5],

            # Disctoninuous subsets of 4
            [0, 1, 3, 4],
            [1, 2, 4, 5],
            [0, 2, 3, 5],

            # All channels
            [0, 1, 2, 3, 4, 5]
        ]

        assert len(self.connections) == 16

        self.convs = nn.ModuleList([
            nn.Conv2d(len(channels), 1, kernel_size=5) # yields oH=10, oW=10 feature maps
                                                       # Note iH=16, iW=16
            for channels in self.connections
        ])

    def forward(self, x):
        outputs = [
            conv(x[:, channels])
            for channels, conv in zip(self.connections, self.convs)
        ]
        return torch.cat(outputs, dim=1)



class LeNet5(nn.Module):
    def __init__(self, use_C3=False):
        super().__init__()

        self.conv_subsample_stack = nn.Sequential(
            nn.Conv2d(1, 6, 5),           # in_channels, out_channels, kernel_size
            Subsample2d(6, 2, stride=2),  # in_channels, kernel_size, then kwargs
            C3() if use_C3 else nn.Conv2d(6, 16, 5),
            Subsample2d(16, 2, stride=2),
            # At this point, the output takes on shape (batches, 16, 5, 5)
            nn.Conv2d(16, 120, 5)
        )

        self.linear = nn.Linear(120, classes)
        # shape of output: (batches, 120, 1, 1)
    def forward(self, x):
        prelin = self.conv_subsample_stack(x)
        return self.linear(prelin.squeeze())
