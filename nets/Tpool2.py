# coding=utf-8
import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 28, 28)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=64,            # n_filters
                kernel_size=(8, 20),              # filter size
                stride=1,                   # filter movement/step
                # padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2, stride=2, dilation=1),    # choose max value in 2x2 area, output shape (16, 14, 14)
            nn.Dropout()
        )
        self.conv2 = nn.Sequential(         # input shape (16, 14, 14)
            nn.Conv2d(64, 64, (4, 10), 1),     # output shape (32, 14, 14)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(1, 1),                # output shape (32, 7, 7)
            nn.Dropout()
        )
        self.out = nn.Linear(26624, 12)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        print(x.shape)
        x = self.conv2(x)
        print(x.shape)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output    # return x for visualization

if __name__ == "__main__":
    model = CNN()
    x = torch.randn((64, 1, 39, 101))
    y = model(x)
    print(y.shape)