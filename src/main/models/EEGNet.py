import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, max_norm=1, **kwargs):
        self.max_norm = max_norm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        self.weight.data = torch.renorm(
            self.weight.data, p=2, dim=0, maxnorm=self.max_norm
        )
        return super(Conv2dWithConstraint, self).forward(x)


class EEGNet(nn.Module):
    def __init__(self,
                 num_classes: int = 4,
                 num_channels: int = 32,
                 num_samples: int = 128,
                 F1: int = 8,
                 F2: int = 16,
                 D: int = 2,
                 kernel_length_1: int = 64,
                 kernel_length_2: int = 16,
                 dropout_rate: float = 0.25,
                 last_dim = 400
                 ):
        super(EEGNet, self).__init__()
        self.num_classes = num_classes
        self.num_channels = num_channels
        self.num_samples = num_samples
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.kernel_length_1 = kernel_length_1
        self.kernel_length_2 = kernel_length_2
        self.dropout_rate = dropout_rate

        self.block1 = nn.Sequential(
            nn.Conv2d(1,
                      self.F1,
                      (1, self.F1),
                      padding=(0, self.kernel_length_1 // 2),
                      bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            Conv2dWithConstraint(self.F1,
                                 self.F1 * self.D,
                                 (self.num_channels, 1),
                                 max_norm=1,
                                 stride=1,
                                 padding=(0, 0),
                                 groups=self.F1,
                                 bias=False),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),
            nn.Dropout(p=dropout_rate)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D,
                      self.F1 * self.D,
                      (1, self.kernel_length_2),
                      stride=1,
                      padding=(0, self.kernel_length_2 // 2),
                      groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D,
                      self.F2,
                      1,
                      padding=(0, 0),
                      groups=1,
                      bias=False,
                      stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(p=dropout_rate)
        )

        self.linear = nn.Linear(last_dim, num_classes, bias=False)

    @property
    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_channels, self.num_samples)

            mock_eeg = self.block1(mock_eeg)
            mock_eeg = self.block2(mock_eeg)

        return mock_eeg.shape[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)

        return x


if __name__ == '__main__':
    from torchsummary import summary
    m = EEGNet(num_classes= 2,
                 num_channels= 62,
                 num_samples= 750)
    summary(m, input_size=(1, 62, 750), batch_size=1, device='cpu')

    from fvcore.nn import FlopCountAnalysis
    inputs = torch.randn(1, 1, 62, 750)
    flop_counter = FlopCountAnalysis(m, inputs)
    print(f"FLOPs: {flop_counter.total()}")