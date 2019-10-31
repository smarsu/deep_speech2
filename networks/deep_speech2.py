# Copyright 2019 smarsu. All Rights Reserved.

"""Implement deep speech2 by pytorch.

Reference:
    Deep Speech 2: End-to-End Speech Recognition in English and Mandarin
    https://arxiv.org/abs/1512.02595
    https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""

import torch


def shape_infer(x, filter_size, stride):
    """conv shape infer."""
    return (x - filter_size) // stride + 1


def calc_downsampled_t_length(t):
    t = shape_infer(t, 5, 4)
    t = shape_infer(t, 5, 2)
    t = shape_infer(t, 5, 2)
    return t


class ClipedRelu(torch.nn.Module):
    def __init__(self, inplace=True, up_border=20):
        super().__init__()
        self.up_border = torch.tensor(up_border, dtype=torch.float).cuda()
        self.relu = torch.nn.ReLU(inplace)


    def forward(self, x):
        x = self.relu(x)
        x = torch.min(x, self.up_border)
        return x


class DeepSpeech2(torch.nn.Module):
    def __init__(self, freq_dim, vocab_size, up_border=20):
        super().__init__()
        self.vocab_size = vocab_size  # black in vocab_size
        c1 = 32
        c2 = 32
        c3 = 96

        self.up_border = torch.tensor(up_border, dtype=torch.float)
        
        self.convblock = torch.nn.Sequential(
        torch.nn.Conv2d(in_channels=1, 
                                     out_channels=c1,
                                     kernel_size=[5, 5],
                                     stride=[4, 4]),  # input: [batch_size, time, freq, 1]
        torch.nn.BatchNorm2d(c1),
        ClipedRelu(inplace=True),
        torch.nn.Conv2d(in_channels=c1, 
                                     out_channels=c2,
                                     kernel_size=[5, 5],
                                     stride=[2, 2]),  # We add stride in time dim to downsample 8 times
        torch.nn.BatchNorm2d(c2),
        ClipedRelu(inplace=True),
        torch.nn.Conv2d(in_channels=c2, 
                                     out_channels=c3,
                                     kernel_size=[5, 5],
                                     stride=[2, 2]),  # We add stride in time dim to downsample 8 times
        torch.nn.BatchNorm2d(c3),
        ClipedRelu(inplace=True),
        )

        freq_dim = shape_infer(freq_dim, 5, 4)
        freq_dim = shape_infer(freq_dim, 5, 2)
        freq_dim = shape_infer(freq_dim, 5, 2)
        self.freq_dim = freq_dim

        self.gru = torch.nn.GRU(input_size=self.freq_dim * c3,
                                hidden_size=1280,
                                num_layers=1,
                                batch_first=True,
                                dropout=0,
                                bidirectional=True)

        self.fc = torch.nn.Linear(2 * 1280, self.vocab_size)
        self.ctc_loss = torch.nn.CTCLoss()

    
    def forward(self, x):
        x = self.convblock(x)

        batch, feat, time, freq = list(x.shape)
        x = x.permute([0, 2, 3, 1])
        x = torch.reshape(x, [batch, time, freq * feat])

        x = self.gru(x)
        x = self.fc(x[0])
        # x = x.permute(1, 0, 2)
        # x = x.log_softmax(2)
        # x = self.ctc_loss(x, target, input_lengths, target_lengths)
        return x


ctc_loss = torch.nn.CTCLoss()


if __name__ == '__main__':
    deep_speech2 = DeepSpeech2(201, 1000)
    while True:
        x = torch.rand(32, 1, 1000, 201)
        target = torch.randint(low=1, high=1000, size=(32, 20))
        input_lengths = torch.full(size=(32,), fill_value=calc_downsampled_t_length(1000), dtype=torch.int)
        target_lengths = torch.full(size=(32,), fill_value=19, dtype=torch.int)
        x = deep_speech2(x)
        x = x.permute(1, 0, 2)
        x = x.log_softmax(2)
        print(x.cpu().detach().numpy().shape)
        x = ctc_loss(x, target, input_lengths, target_lengths)
        print(x.cpu().detach().numpy())
        print(x.cpu().detach().numpy().shape)
