import torch


class ShallowSpeech(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Conv2d(inchannels=96*2,
                                    out_channels=96*2,
                                    kernel_size=[1, 3],
                                    stride=[1, 2])

        self.gru = torch.nn.GRU(input_size=96 * 2,
                                hidden_size=96,
                                num_layers=1,
                                batch_first=True,
                                dropout=0,
                                bidirectional=True)

        self.fc = torch.nn.Linear(2 * 96, 4231)


    def forward(self, x):
        for _ in range(5):
            x = self.conv(x)

        x = torch.squeeze(x)  # [32, 92 * 2, 16000 * 10 * 2 // 96 // 2]
        x = torch.transpose(x, [0, 2, 1])

        x = self.gru(x)
        x = self.fc(x[0])
        return x


if __name__ == '__main__':
    shallow_speech = ShallowSpeech().cuda()
    while True:
        x = torch.rand(32, 96 * 2, 1, 16000 * 10 * 2 // 96 // 2).cuda()
        x = shallow_speech(x)
        print(x.cpu().detach().numpy().shape)
