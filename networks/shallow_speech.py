import torch


class ShallowSpeech(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_channels=96*2,
                                    out_channels=96*2,
                                    kernel_size=[1, 3],
                                    stride=[1, 2],
                                    padding=[0, 1])
        self.batch_norm = torch.nn.BatchNorm2d(num_features=96*2)


        self.gru = torch.nn.GRU(input_size=96 * 2,
                                hidden_size=96,
                                num_layers=1,
                                batch_first=True,
                                dropout=0,
                                bidirectional=True)

        self.fc = torch.nn.Linear(2 * 96, 4231)


    def forward(self, x):
        x = x.permute([0, 3, 1, 2])

        for _ in range(5):
            x = self.conv(x)
            x = self.batch_norm(x)
            x = torch.relu(x)

        x = torch.squeeze(x)  # [32, 92 * 2, 16000 * 10 * 2 // 96 // 2]
        x = x.permute([0, 2, 1])

        x = self.gru(x)
        x = self.fc(x[0])
        return x


if __name__ == '__main__':
    shallow_speech = ShallowSpeech().cuda()
    while True:
        x = torch.rand(32, 1, 16000 * 10 * 2 // 96 // 2, 96 * 2).cuda()
        x = shallow_speech(x)
        print(x.cpu().detach().numpy().shape)
