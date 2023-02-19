import torch.nn as nn
import torch



# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False, pooling=False):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsampleOrNot = downsample
        self.pooling = pooling
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsampleOrNot:
            residual = self.downsample(x)
            out += residual
        if self.pooling:
            out = self.maxpool(out)
        out = self.dropout(out)
        return out


def torch_stft(X_train, hop_length_mult=4):
    signal = []
    for s in range(X_train.shape[1]):
        spectral = torch.stft(X_train[:, s, :],
                              n_fft=256,
                              hop_length=256 * 1 // hop_length_mult,
                              center=False,
                              onesided=True)
        signal.append(spectral)

    signal1 = torch.stack(signal)[:, :, :, :, 0].permute(1, 0, 2, 3)
    signal2 = torch.stack(signal)[:, :, :, :, 1].permute(1, 0, 2, 3)

    return torch.cat([torch.log(torch.abs(signal1) + 1e-8), torch.log(torch.abs(signal2) + 1e-8)], dim=1)


class CNNEncoder2D_ISRUC(nn.Module):
    def __init__(self, n_dim=128, embedding_size=96, nclass=5):
        super(CNNEncoder2D_ISRUC, self).__init__()

        base_channels=6

        self.avg_pool = torch.nn.AvgPool1d(2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(2*base_channels, 3*base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3*base_channels),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(3*base_channels, 4*base_channels, 2, True, False)
        self.conv3 = ResBlock(4*base_channels, 8*base_channels, 2, True, True)
        self.conv4 = ResBlock(8*base_channels, 16*base_channels, 2, True, True)
        self.n_dim = n_dim

        self.sup = nn.Sequential(
            nn.Linear(64*base_channels, 16*base_channels, bias=True),
            nn.ReLU(),
            nn.Linear(embedding_size, 5, bias=True),
        )

    def get_readout_layer(self):
        return self.sup[2]

    def forward(self, x, embed_only=False):
        x = self.avg_pool(x)
        x = torch_stft(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape(x.shape[0], -1)
        if embed_only:
            return self.sup[1](self.sup[0](x))
        return self.sup(x)

class CNNEncoder2D_IIIC(nn.Module):
    def __init__(self, n_dim=128, embedding_size=32):
        super(CNNEncoder2D_IIIC, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(32, 32, 2, True, False)
        self.conv3 = ResBlock(32, 64, 2, True, True)
        self.conv4 = ResBlock(64, 64, 2, True, True)
        self.n_dim = n_dim

        self.sup = nn.Sequential(
            nn.Linear(512, embedding_size, bias=True),
            nn.ReLU(),
            nn.Linear(embedding_size, 6, bias=True),
        )


    def get_readout_layer(self):
        return self.sup[2]

    def forward(self, x, embed_only=False):
        x = torch_stft(x, hop_length_mult=8)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape(x.shape[0], -1)
        if embed_only:
            return self.sup[1](self.sup[0](x))
        return self.sup(x)


class CNNEncoder2D_IIIC2(nn.Module):
    def __init__(self, n_dim=128, embedding_size=48, nclass=6):
        super(CNNEncoder2D_IIIC2, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(inplace=True),
        )
        self.conv2 = ResBlock(32, 32, 2, True, False)
        self.conv3 = ResBlock(32, 64, 2, True, True)
        self.conv4 = ResBlock(64, 64, 2, True, True)
        self.n_dim = n_dim

        self.sup = nn.Sequential(
            nn.Linear(512, 64, bias=True),
            nn.ReLU(),
            nn.Linear(64, embedding_size, bias=True),
            nn.ReLU(),
        )
        self.fc = nn.Linear(embedding_size, 6, bias=True)


    def get_readout_layer(self):
        return self.fc

    def forward(self, x, embed_only=False):
        x = torch_stft(x, hop_length_mult=8)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape(x.shape[0], -1)
        x = self.sup(x)
        if embed_only:
            return x
        return self.fc(x)