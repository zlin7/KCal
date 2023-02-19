import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeAttn(nn.Module):
    def __init__(self, input_features, attn_dim):
        """
        This is the general knowledge-guided attention module.
        It will transform the input and knowledge with 2 linear layers, computes attention, and then aggregate.
        :param input_features: the number of features for each
        :param attn_dim: the number of hidden nodes in the attention mechanism
        """
        super(KnowledgeAttn, self).__init__()
        self.input_features = input_features
        self.attn_dim = attn_dim
        self.n_knowledge = 1

        self.att_W = nn.Linear(self.input_features + self.n_knowledge, self.attn_dim, bias=False)
        self.att_v = nn.Linear(self.attn_dim, 1, bias=False)

        self.init()

    def init(self):
        nn.init.normal_(self.att_W.weight)
        nn.init.normal_(self.att_v.weight)

    @classmethod
    def attention_sum(cls, x, attn):
        """
        :param x: of shape (-1, D, nfeatures)
        :param attn: of shape (-1, D, 1)
        """
        return torch.sum(torch.mul(attn, x), 1)


    def forward(self, x, k):
        """
        :param x: shape of (-1, D, input_features)
        :param k: shape of (-1, D, 1)
        :return:
            out: shape of (-1, input_features), the aggregated x
            attn: shape of (-1, D, 1)
        """
        tmp = torch.cat([x, k], dim=-1)
        e = self.att_v(torch.tanh(self.att_W(tmp)))
        attn = F.softmax(e, 1)
        out = self.attention_sum(x, attn)
        return out, attn

#============================================================

class BeatNet(nn.Module):
    #Attention for the CNN step/ beat level/local information
    def __init__(self, n=3000, T=50,
                 conv_out_channels=64):
        """
        :param n: size of each 10-second-data
        :param T: size of each smaller segment used to capture local information in the CNN stage
        :param conv_out_channels: also called number of filters/kernels
        """
        super(BeatNet, self).__init__()
        self.n, self.M, self.T = n, int(n/T), T
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = 32
        self.conv_stride = 2
        #Define conv and conv_k, the two Conv1d modules
        self.conv = nn.Conv1d(in_channels=1,
                              out_channels=self.conv_out_channels,
                              kernel_size=self.conv_kernel_size,
                              stride=self.conv_stride)

        self.conv_k = nn.Conv1d(in_channels=1,
                                out_channels=1,
                                kernel_size=self.conv_kernel_size,
                                stride=self.conv_stride)

        self.att_cnn_dim = 8
        #Define attn, the KnowledgeAttn module
        self.attn = KnowledgeAttn(self.conv_out_channels, self.att_cnn_dim)

    def forward(self, x, k_beat):
        """
        :param x: shape (batch, n)
        :param k_beat: shape (batch, n)
        :return:
            out: shape (batch, M, self.conv_out_channels)
            alpha: shape (batch * M, N, 1) where N is a result of convolution
        """
        x = x.view(-1, self.T).unsqueeze(1)
        k_beat = k_beat.view(-1, self.T).unsqueeze(1)
        x = F.relu(self.conv(x))  # Here number of filters K=64
        k_beat = F.relu(self.conv_k(k_beat))  # Conv1d(1, 1, kernel_size=(32,), stride=(2,)) => k_beat:[128*60,1,10].

        x = x.permute(0, 2, 1)  # x:[128*60,10,64]
        k_beat = k_beat.permute(0, 2, 1)
        out, alpha = self.attn(x, k_beat)
        out = out.view(-1, self.M, self.conv_out_channels)
        return out, alpha


class RhythmNet(nn.Module):
    def __init__(self, n=3000, T=50, input_size=64, rhythm_out_size=8):
        """
        :param n: size of each 10-second-data
        :param T: size of each smaller segment used to capture local information in the CNN stage
        :param input_size: This is the same as the # of filters/kernels in the CNN part.
        :param rhythm_out_size: output size of this netowrk
        """
        #input_size is the cnn_out_channels
        super(RhythmNet, self).__init__()
        self.n, self.M, self.T = n, int(n/T), T
        self.input_size = input_size

        self.rnn_hidden_size = 32
        ### define lstm: LSTM Input is of shape (batch size, M, input_size)
        self.lstm = nn.LSTM(input_size=self.input_size, #self.conv_out_channels,
                            hidden_size=self.rnn_hidden_size,
                            num_layers=1, batch_first=True, bidirectional=True)

        ### Attention mechanism: define attn to be a KnowledgeAttn
        self.att_rnn_dim = 8
        self.attn = KnowledgeAttn(2 * self.rnn_hidden_size, self.att_rnn_dim)

        ### Define the Dropout and fully connecte layers (fc and do)
        self.out_size = rhythm_out_size
        self.fc = nn.Linear(2 * self.rnn_hidden_size, self.out_size)
        self.do = nn.Dropout(p=0.5)



    def forward(self, x, k_rhythm):
        """
        :param x: shape (batch, M, self.input_size)
        :param k_rhythm: shape (batch, M)
        :return:
            out: shape (batch, self.out_size)
            beta: shape (batch, M, 1)
        """

        ### reshape for rnn
        k_rhythm = k_rhythm.unsqueeze(-1)  # [128, 60, 1]
        ### rnn
        o, (ht, ct) = self.lstm(x)  # o:[batch,60,64] (in the paper this is called h

        x, beta = self.attn(o, k_rhythm)
        ### fc and Dropout
        x = F.relu(self.fc(x))  # [128, 64->8]
        out = self.do(x)
        return out, beta

class FreqNet(nn.Module):
    def __init__(self, n_channels=4, n=3000, T=50, embedding_size=8, nclass = 4):
        """
        :param n_channels: number of channels (F in the paper). We will need to define this many BeatNet & RhythmNet nets.
        :param n: size of each 10-second-data
        :param T: size of each smaller segment used to capture local information in the CNN stage
        """
        super(FreqNet, self).__init__()
        self.n, self.M, self.T = n, int(n / T), T
        self.n_class = nclass
        self.n_channels = n_channels
        self.conv_out_channels=64
        self.rhythm_out_size=embedding_size

        self.beat_nets = nn.ModuleList()
        self.rhythm_nets = nn.ModuleList()
        #use self.beat_nets.append() and self.rhythm_nets.append() to append 4 BeatNets/RhythmNets
        for channel_i in range(self.n_channels):
            self.beat_nets.append(BeatNet(self.n, self.T, self.conv_out_channels))
            self.rhythm_nets.append(RhythmNet(self.n, self.T, self.conv_out_channels, self.rhythm_out_size))


        self.att_channel_dim = 2
        ### Add the frequency attention module using KnowledgeAttn (attn)
        self.attn = KnowledgeAttn(self.rhythm_out_size, self.att_channel_dim)

        ### Create the fully-connected output layer (fc)
        self.fc = nn.Linear(self.rhythm_out_size, self.n_class)

    def get_readout_layer(self):
        return self.fc

    def forward(self, data_tuple_, embed_only=False):
        """
        We need to use the attention submodules to process data from each channel separately, and then pass the
            output through an attention on frequency for the final output

        :param x: shape (n_channels, batch, n)
        :param k_beats: (n_channels, batch, n)
        :param k_rhythms: (n_channels, batch, M)
        :param k_freq: (n_channels, batch, 1)
        :return:
            out: softmax output for each data point, shpae (batch, n_class)
            gama: the attention value on channels
        """
        x, k_beats, k_rhythms, k_freq = data_tuple_
        #x, k_beats, k_rhythms, k_freq = [_.permute(1,0,2) for _ in data_tuple_]
        new_x = [None for _ in range(self.n_channels)]
        for i in range(self.n_channels):
            tx, _ = self.beat_nets[i](x[i], k_beats[i])
            new_x[i], _ = self.rhythm_nets[i](tx, k_rhythms[i])
        x = torch.stack(new_x, 1)  # [128,8] -> [128,4,8]

        # ### attention on channel
        k_freq = k_freq.permute(1, 0, 2) #[4,128,1] -> [128,4,1]
        x, gama = self.attn(x, k_freq)

        ### fc
        #out = F.softmax(self.fc(x), 1) #CrossEntropy expects unnormalized scores.
        if embed_only: return x
        out = self.fc(x)
        return out