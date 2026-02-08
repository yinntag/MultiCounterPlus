import torch.nn as nn
import torch


class ThreeLayerLSTM(nn.Module):
    def __init__(self):
        super(ThreeLayerLSTM, self).__init__()

        self.lstm1 = nn.LSTM(256, 512, batch_first=True)
        self.lstm2 = nn.LSTM(512, 512, batch_first=True)
        self.lstm3 = nn.LSTM(512, 256, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out, _ = self.lstm2(out)
        out, _ = self.lstm3(out)

        return out


# if __name__ == '__main__':
#     model = ThreeLayerLSTM()
#     x = torch.randn(2, 64, 256)
#     out = model(x)
#     print(out.shape)

