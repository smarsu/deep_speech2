import torch


class ShallowSpeech(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = torch.nn.GRU(input_size=96 * 2,
                                hidden_size=96,
                                num_layers=1,
                                batch_first=True,
                                dropout=0,
                                bidirectional=True)

        self.fc = torch.nn.Linear(2 * 96, 4231)


    def forward(self, x):
        x = self.gru(x)
        x = self.fc(x[0])


if __name__ == '__main__':
    shallow_speech = ShallowSpeech().cuda()
    while True:
        x = torch.rand(32, 16000 * 10 * 2 // 96 // 2, 96 * 2).cuda()
        x = shallow_speech(x)
        print(x.cpu().detach().numpy().shape)
