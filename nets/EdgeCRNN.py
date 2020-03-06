# coding=utf-8
import torch
import torch.nn as nn


def frist_conv(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, 1, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)  # nn.Relu()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        # nn.PReLU()
        nn.ReLU(inplace=True)
    )


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()  # (batchsize, channels_per_group, groups, height, width)

    # flatten
    x = x.view(batchsize, -1, height, width)  # (batchsize, -1, height, width)

    return x


def Base_block(oup_inc, stride):

    banch = nn.Sequential(
        # pw
        nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup_inc),
        nn.ReLU(inplace=True),
        # dw
        nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
        nn.BatchNorm2d(oup_inc),
        # pw-linear
        nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup_inc),
        nn.ReLU(inplace=True),
    )
    return banch


def EdgeCRNN_block(inp, oup_inc, stride):
    left_banch = nn.Sequential(
        # dw
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        # pw-linear
        nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup_inc),
        nn.ReLU(inplace=True),
    )
    right_banch = nn.Sequential(
                # pw
                nn.Conv2d(inp, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(oup_inc, oup_inc, 3, stride, 1, groups=oup_inc, bias=False),
                nn.BatchNorm2d(oup_inc),
                # pw-linear
                nn.Conv2d(oup_inc, oup_inc, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup_inc),
                nn.ReLU(inplace=True),
            )
    return left_banch, right_banch


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, benchmodel):
        super(InvertedResidual, self).__init__()
        self.benchmodel = benchmodel
        self.stride = stride
        assert stride in [1, 2]

        oup_inc = oup // 2

        if self.benchmodel == 1:
            # assert inp == oup_inc
            self.banch2 = Base_block(oup_inc, stride)
        else:
            self.banch1, self.banch2 = EdgeCRNN_block(inp, oup_inc, stride)

    @staticmethod
    def _concat(x, out):
        # concatenate along channel axis
        return torch.cat((x, out), 1)

    def forward(self, x):
        if 1 == self.benchmodel:
            x1 = torch.chunk(x, 2, 1)[0]
            x2 = torch.chunk(x, 2, 1)[1]
            out = self._concat(x1, self.banch2(x2))
        elif 2 == self.benchmodel:
            out = self._concat(self.banch1(x), self.banch2(x))

        return channel_shuffle(out, 2)


class EdgeCRNN(nn.Module):
    def __init__(self, n_class=12, input_size=101, width_mult=1.):
        super(EdgeCRNN, self).__init__()

        # assert input_size % 32 == 0

        self.stage_repeats = [2, 3, 2]
        # index 0 is invalid and should never be called.
        # only used for indexing convenience.
        if width_mult == 0.5:
            self.stage_out_channels = [-1, 16, 32, 64, 128, 256]  # *2  *2  16,  32,  64, 128, 256
        elif width_mult == 1.0:
            self.stage_out_channels = [-1, 24, 72, 144, 288, 512]  # *4.9 *2  24, 72, 144, 288, 512
        elif width_mult == 1.5:
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]  # *7.3 *2
        elif width_mult == 2.0:
            self.stage_out_channels = [-1, 24, 160, 320, 640, 1024]  # *9.3  *2
        else:
            raise ValueError(
                """groups is not supported for
                       1x1 Grouped Convolutions""")
        # building first layer
        input_channel = self.stage_out_channels[1]
        self.conv1 = frist_conv(1, input_channel)  # 1 dim
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.features = []
        # building Stage2-4
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]
            for i in range(numrepeat):
                if i == 0:
                    # inp, oup, stride, benchmodel):
                    self.features.append(InvertedResidual(input_channel, output_channel, 2, 2))
                else:
                    self.features.append(InvertedResidual(input_channel, output_channel, 1, 1))
                input_channel = output_channel
        # make it nn.Sequential
        self.features = nn.Sequential(*self.features)  # 16层网络
        # building last several layers
        self.conv_last = conv_1x1_bn(input_channel, self.stage_out_channels[-1])

        self.globalpool = nn.Sequential(nn.AvgPool2d((3, 1), stride=(1, 1)))  # rnn->cnn (3,1)->(3, 7)
        # first-layer(3,1),other(2,1)； cnn first（3,7），other（2,4）

        # add RNN block
        self.hidden_size = 64
        # self.RNN = nn.RNN(self.stage_out_channels[-1], self.hidden_size, num_layers=1, batch_first=True)
        self.RNN = nn.LSTM(self.stage_out_channels[-1], self.hidden_size, num_layers=1, batch_first=True)
        # self.RNN = nn.GRU(self.stage_out_channels[-1], self.hidden_size, num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(self.hidden_size, n_class))

        # building classifier CNN
        # self.classifier = nn.Sequential(nn.Linear(self.stage_out_channels[-1], n_class))

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        # print(x.shape)
        x = self.globalpool(x)  # shape(64,1024,1,4)

        # CNN
        # x = x.squeeze()
        # x = x.view(-1, self.stage_out_channels[-1])

        # add RNN block
        x = x.squeeze(dim=2).permute(0, 2, 1)  # shape(64,1024,1,4)--> shape(b, w, c)  (64, 7, 1024)
        self.RNN.flatten_parameters()
        x, _ = self.RNN(x)  # shape(64, 7, 1024)
        x = x.permute(0, 2, 1).mean(2)  # shape(1, 64,1024)--> (64,1024, 7)

        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.randn((64, 1, 39, 101))  # (b, c, h, w)
    model = EdgeCRNN(width_mult=1)
    x = model(x)
    print(x.shape)
