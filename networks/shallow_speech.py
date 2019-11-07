import torch
import math


class ShallowSpeech(torch.nn.Module):
    def __init__(self, 
                 c_in=417,
                 c1=512,
                 c2=1024,
                 c3=2048,
                 crnn=1024,
                 co=404):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(in_channels=c_in, 
                                     out_channels=c1, 
                                     kernel_size=(3, 1), 
                                     stride=1, 
                                     padding=0, 
                                     dilation=1, 
                                     groups=1, 
                                     bias=True)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=(2, 1), 
                                           stride=(2, 1), 
                                           padding=0, 
                                           dilation=1, 
                                           return_indices=False, 
                                           ceil_mode=True)  # ceil nor floor
        self.conv2 = torch.nn.Conv2d(in_channels=c1, 
                                     out_channels=c2, 
                                     kernel_size=(3, 1), 
                                     stride=1, 
                                     padding=0, 
                                     dilation=1, 
                                     groups=1, 
                                     bias=True)
        self.conv3 = torch.nn.Conv2d(in_channels=c2, 
                                     out_channels=c3, 
                                     kernel_size=(3, 1), 
                                     stride=1, 
                                     padding=0, 
                                     dilation=1, 
                                     groups=1, 
                                     bias=True)
        # self.conv4 = torch.nn.Conv2d(in_channels=c3, 
        #                              out_channels=co, 
        #                              kernel_size=(1, 1), 
        #                              stride=1, 
        #                              padding=0, 
        #                              dilation=1, 
        #                              groups=1, 
        #                              bias=True)
        self.lstm = torch.nn.LSTM(input_size=c3,
                                  hidden_size=crnn,
                                  num_layers=2,
                                  batch_first=True,
                                  dropout=0,  # god dropout
                                  bidirectional=True)
        self.fc = torch.nn.Linear(crnn * 2, co)
        

    def calc_t_length(self, t):
        t = t - 2  # conv1
        t = math.ceil(t / 2)  # pool1
        t = t - 2  # conv2
        t = t - 2  # conv3
        return t


    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)

        x = self.maxpool1(x)

        x = self.conv2(x)
        x = torch.relu(x)

        x = self.conv3(x)
        x = torch.relu(x)

        # x = self.conv4(x)   # [n, c, t, 1]

        # x = x[:, :, :, 0]
        # x = x.permute([0, 2, 1])

        x = x[:, :, :, 0]
        x = x.permute([0, 2, 1])

        x = self.lstm(x)
        x = self.fc(x[0])
        return x


if __name__ == '__main__':
    pass
