# coding=utf-8
# ref： https://github.com/tonylins/pytorch-mobilenet-v2/blob/master/MobileNetV2.py
import torch
import torch.nn as nn
import math


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def last_conv_1x1_bn(inp, oup, kernel):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)  # add

        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, n_class=12, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 2],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        assert input_size % 32 == 0
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        self.features = [conv_bn(1, input_channel, 2)]  # the first conv 1
        # building inverted residual blocks
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                if i == 0:
                    self.features.append(block(input_channel, output_channel, s, expand_ratio=t))
                else:
                    self.features.append(block(input_channel, output_channel, 1, expand_ratio=t))
                input_channel = output_channel
        # self.input_channel = input_channel
        # building last several layers
        self.features.append(conv_1x1_bn(input_channel, self.last_channel))
        # self.features.append(last_conv_1x1_bn(input_channel, self.last_channel, (3, 1)))  # RNN

        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)

        # self.hidden_size = 256
        # add RNN block
        # self.RNN = nn.RNN(last_channel, last_channel, num_layers=1, batch_first=True, nonlinearity="relu")
        # self.RNN = nn.LSTM(last_channel, self.last_channel, num_layers=1, batch_first=True)
        # self.RNN = nn.GRU(last_channel, last_channel, num_layers=1, batch_first=True)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)  # shape(1,1280,1,7)  (batch. channel, h, w)
        print(x.shape)
        x = x.mean(3).mean(2)  # (batch. channel, h, w) -->(b, c6)
        print(x.shape)

        # add RNN block
        # x = x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (batch, c, h, w)-->(b, w, c)
        # x = x.squeeze(dim=2).permute(0, 2, 1)  # (batch, c, h, w)-->(b, w, c)
        # x, _ = self.RNN(x)
        # x = x.permute(0, 2, 1).mean(2)

        x = self.classifier(x)  # input shape(1,1280)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == "__main__":
    x = torch.randn(64, 1, 39, 101)
    model = MobileNetV2(width_mult=1)
    model.train()
    import time
    # print(time.time())
    x = model(x)
    # print(time.time())
    # print(x.shape)
    print(model)