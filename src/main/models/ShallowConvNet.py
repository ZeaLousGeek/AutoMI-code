import torch
import torch.nn as nn

from src.main.train.utils.results import print_model_summary


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


def initialize_weight(model, method):
    method = dict(normal=['normal_', dict(mean=0, std=0.01)],
                  xavier_uni=['xavier_uniform_', dict()],
                  xavier_normal=['xavier_normal_', dict()],
                  he_uni=['kaiming_uniform_', dict()],
                  he_normal=['kaiming_normal_', dict()]).get(method)
    if method is None:
        return None

    for module in model.modules():
        if module.__class__.__name__ in ['LSTM']:
            for param in module._all_weights[0]:
                if param.startswith('weight'):
                    getattr(nn.init, method[0])(getattr(module, param), **method[1])
                elif param.startswith('bias'):
                    nn.init.constant_(getattr(module, param), 0)
        else:
            if hasattr(module, "weight"):
                if not ("BatchNorm" in module.__class__.__name__):
                    getattr(nn.init, method[0])(module.weight, **method[1])
                else:
                    nn.init.constant_(module.weight, 1)
                if hasattr(module, "bias"):
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)


class ShallowConvNet(nn.Module):
    def __init__(
            self,
            n_classes,
            ch_nums=22,
            F1=None,
            T1=None,
            F2=None,
            P1_T=None,
            P1_S=None,
            drop_out=None,
            pool_mode=None,
            weight_init_method=None,
            last_dim=None,
    ):
        super(ShallowConvNet, self).__init__()
        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, T1)),
            nn.Conv2d(F1, F2, (ch_nums, 1), bias=False),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer((1, P1_T), (1, P1_S)),
            ActLog(),
            nn.Dropout(drop_out),
            nn.Flatten(),
            nn.Linear(last_dim, n_classes)
        )

        initialize_weight(self, weight_init_method)

    def forward(self, x):
        out = self.net(x)
        return out


class ShallowConvNetWithConstraint(nn.Module):
    def __init__(
            self,
            n_classes,
            ch_nums=22,
            F1=None,
            T1=None,
            F2=None,
            P1_T=None,
            P1_S=None,
            drop_out=None,
            pool_mode=None,
            weight_init_method=None,
            last_dim=None,
    ):
        super(ShallowConvNet, self).__init__()
        pooling_layer = dict(max=nn.MaxPool2d, mean=nn.AvgPool2d)[pool_mode]
        self.net = nn.Sequential(
            Conv2dWithConstraint(1, F1, (1, T1), max_norm=2),
            Conv2dWithConstraint(F1, F2, (ch_nums, 1), bias=False, max_norm=2),
            nn.BatchNorm2d(F2),
            ActSquare(),
            pooling_layer((1, P1_T), (1, P1_S)),
            ActLog(),
            nn.Dropout(drop_out),
            nn.Flatten(),
            LinearWithConstraint(last_dim, n_classes, max_norm=0.5)
        )

        initialize_weight(self, weight_init_method)

    def forward(self, x):
        out = self.net(x)
        return out


class ActSquare(nn.Module):
    def __init__(self):
        super(ActSquare, self).__init__()
        pass

    def forward(self, x):
        return torch.square(x)


class ActLog(nn.Module):
    def __init__(self, eps=1e-06):
        super(ActLog, self).__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


def test():
    pass

def main(chs, cls):
    from torchsummary import summary
    m = ShallowConvNet(n_classes=cls,
                       ch_nums=chs,
                       F1=40,
                       T1=25,
                       F2=40,
                       P1_T=75,
                       P1_S=15,
                       drop_out=0.5,
                       pool_mode="max",
                       weight_init_method='normal_',
                       last_dim=1760)
    summary(m, input_size=(1, chs, 750), batch_size=1, device='cpu')

    from fvcore.nn import FlopCountAnalysis
    inputs = torch.randn(1, 1, chs, 750)
    flop_counter = FlopCountAnalysis(m, inputs)
    print(f"FLOPs: {flop_counter.total()}")

if __name__ == '__main__':
    chs = 62
    cls = 2
    main(chs, cls)
    print_model_summary(model=ShallowConvNet(n_classes=cls,
                       ch_nums=chs,
                       F1=40,
                       T1=25,
                       F2=40,
                       P1_T=75,
                       P1_S=15,
                       drop_out=0.5,
                       pool_mode="max",
                       weight_init_method='normal_',
                       last_dim=1760),
                    input=torch.randn(1, chs, 750))

