import torch
from torch import nn
from .lstm_layers import ConvLSTM


class PatchDiscriminator(nn.Module):
    '''
        Patch based discriminator which predicts
        the probability whether a subset cube of the image
        is real or fake.
    '''

    def __init__(self, input_channels, nlayers=3, nfilt=16, dropout=0.25, activation='leakyrelu'):
        super(PatchDiscriminator, self).__init__()

        # the first convolution goes from the input channels
        # to the first number of filters
        prev_filt = input_channels
        next_filt = nfilt

        kernel_size = 3

        if activation == 'leakyrelu':
            activation = nn.LeakyReLU(0.2, True)
        elif activation == 'tanh':
            activation = nn.Tanh()

        layers = []
        for i in range(nlayers):
            # for each layer, we apply conv, act and normalization
            layers.append(nn.Conv3d(prev_filt, next_filt, (1, kernel_size, kernel_size),
                                    stride=(1, 2, 2), padding=0))
            layers.append(nn.InstanceNorm3d(next_filt))
            layers.append(activation)

            # the number of filters exponentially increase
            prev_filt = next_filt
            next_filt = nfilt * min([2**i, 8])

        layers += [
            nn.Dropout(dropout),
            nn.Conv3d(prev_filt, next_filt, padding=0,
                      kernel_size=(1, kernel_size, kernel_size), stride=1),
            nn.InstanceNorm3d(next_filt),
            activation
        ]

        # last predictive layer
        layers += [nn.Conv3d(next_filt, 1, (1, kernel_size, kernel_size),
                             padding=0), nn.Sigmoid()]

        self.discriminator = nn.Sequential(*layers)

    def forward(self, x):
        # switch channel and time axis before doing the 3d conv
        return self.discriminator(torch.swapaxes(x, 1, 2))


class LSTMDiscriminator(nn.Module):
    '''
        Patch based discriminator which predicts
        the probability whether a subset cube of the image
        is real or fake.
    '''

    def __init__(self, input_channels, nlayers=3,
                 nfilt=16, inp_size=(96, 96), dropout=0.25):
        super(LSTMDiscriminator, self).__init__()

        # the first convolution goes from the input channels
        # to the first number of filters
        prev_filt = input_channels

        hidden = [nfilt]
        for i in range(1, nlayers + 1):
            hidden.append(hidden[-1] * min([2**i, 8]))

        hidden.append(hidden[-1])

        layers = []
        for i in range(nlayers):
            # for each layer, we apply conv, act and normalization
            # layers.append(nn.ConvLSTM(prev_filt, next_filt, (1, 3, 3),
            #                        stride=(1, 1, 1),
            #                        padding=(0, 1, 1)))
            # layers.append(nn.Tanh())
            # layers.append(nn.BatchNorm3d(next_filt))
            # layers.append(nn.MaxPool3d((1, 2, 2)))
            layers.append(ConvLSTM(prev_filt, hidden[i],
                                   hidden[i + 1], (3, 3), (2, 2)))

            # update the filter value for the next iteration
            # if dropout > 0:
            #    layers.append(nn.Dropout(dropout))

            prev_filt = hidden[i]

            # the number of filters exponentially increase
            inp_size = [int(x / 2) for x in inp_size]

        pred_layers = []
        # last predictive layer
        pred_layers.append(nn.Linear(prev_filt * inp_size[0] * inp_size[1], 128))
        pred_layers.append(nn.LeakyReLU(0.2))

        pred_layers.append(nn.Linear(128, 1))
        pred_layers.append(nn.Sigmoid())

        self.disc_downsample = nn.ModuleList(layers)
        self.disc_prediction = nn.Sequential(*pred_layers)

    def forward(self, x):
        # switch channel and time axis before doing the 3d conv
        c = None
        for layer in self.disc_downsample:
            if c is None:
                hidden_state = None
            else:
                hidden_state = [None, c]
            x, c = layer(x, hidden_state)

        x = x.reshape((x.shape[0], x.shape[1], -1))

        return self.disc_prediction(x)


class ConvDiscriminator(nn.Module):
    '''
        Patch based discriminator which predicts
        the probability whether a subset cube of the image
        is real or fake.
    '''

    def __init__(self, input_channels, nlayers=3,
                 nfilt=16, inp_size=(96, 96), bottleneck_dim=256, dropout=0.25):
        super(ConvDiscriminator, self).__init__()

        # the first convolution goes from the input channels
        # to the first number of filters
        prev_filt = input_channels
        next_filt = nfilt

        hidden = [nfilt]
        for i in range(1, nlayers + 1):
            hidden.append(hidden[-1] * min([2**i, 8]))

        hidden.append(hidden[-1])

        layers = []
        for i in range(nlayers):
            # for each layer, we apply conv, act and normalization
            layers.append(nn.Conv3d(prev_filt, next_filt, (1, 3, 3),
                                    stride=(1, 1, 1),
                                    padding=(0, 1, 1)))
            layers.append(nn.Tanh())
            layers.append(nn.BatchNorm3d(next_filt))
            layers.append(nn.MaxPool3d((1, 2, 2)))
            # layers.append(ConvLSTM(prev_filt, hidden[i],
            #                               hidden[i + 1], (3, 3), (2, 2)))

            # update the filter value for the next iteration
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_filt = next_filt
            next_filt = next_filt * min([2**i, 8])

            # the number of filters exponentially increase
            inp_size = [int(x / 2) for x in inp_size]

        pred_layers = []
        # last predictive layer
        pred_layers.append(nn.Linear(prev_filt * inp_size[0] * inp_size[1], bottleneck_dim))
        pred_layers.append(nn.Tanh())

        pred_layers.append(nn.Linear(bottleneck_dim, 1))
        pred_layers.append(nn.Sigmoid())

        self.disc_downsample = nn.Sequential(*layers)
        self.disc_prediction = nn.Sequential(*pred_layers)

    def forward(self, x):
        # switch channel and time axis before doing the 3d conv
        x = torch.swapaxes(self.disc_downsample(torch.swapaxes(x, 1, 2)), 1, 2)

        x = x.reshape((x.shape[0], x.shape[1], -1))

        return self.disc_prediction(x)
