import torch


class ShallowSpeech(torch.nn.Module):
    def __init__(self, freq_size=201):
        super().__init__()

        self.firstconv = torch.nn.Conv2d(in_channels=1,
                                    out_channels=32,
                                    kernel_size=[3, 3],
                                    stride=[2, 2],
                                    padding=[1, 1])

        self.conv = torch.nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=[3, 3],
                                    stride=[2, 2],
                                    padding=[1, 1])
        self.batch_norm = torch.nn.BatchNorm2d(num_features=32)

        self.gru = torch.nn.GRU(input_size=32*self.calc_t_length(freq_size),
                                hidden_size=512,
                                num_layers=1,
                                batch_first=True,
                                dropout=0,
                                bidirectional=True)

        self.fc = torch.nn.Linear(2 * 512, 4231)

    
    def calc_t_length(self, t):
        for _ in range(4):
            t = (t + 2 * 1 - 2 - 1) // 2 + 1
        return t


    def forward(self, x):
        # x = x.permute([0, 3, 1, 2])

        x = self.firstconv(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        for _ in range(3):
            x = self.conv(x)
            x = self.batch_norm(x)
            x = torch.relu(x)

        batch, feat, time, freq = list(x.shape)
        # x = torch.squeeze(x, 2)  # [32, 92 * 2, 16000 * 10 * 2 // 96 // 2]
        x = x.permute([0, 2, 1, 3])
        x = x.reshape(batch, time, feat * freq)

        x = self.gru(x)
        x = self.fc(x[0])
        return x


if __name__ == '__main__':
    shallow_speech = ShallowSpeech().cuda()
    while True:
        x = torch.rand(32, 1, 16000 * 10 * 2 // 96 // 2, 96 * 2).cuda()
        x = shallow_speech(x)
        print(x.cpu().detach().numpy().shape)
