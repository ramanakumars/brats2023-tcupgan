import torch.nn as nn
import torch
from .lstm_layers import (ConvLSTM, ConvTransposeLSTM)
from .transfer import Transferable
from einops.layers.torch import Rearrange


class LSTMUNet(nn.Module, Transferable):
    gen_type = 'UNet'

    def __init__(self, hidden_dims=[8, 16, 32], decoder_hidden_dims=None, input_channels=1, output_channels=4):
        super(LSTMUNet, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.hidden_dims = hidden_dims
        prev_filt = input_channels

        # invert the hidden layer filters for the decoder
        # also add the input channel at the end
        reverse_encoder_dims = [*hidden_dims[::-1][1:], output_channels]
        if decoder_hidden_dims is None:
            decoder_hidden_dims = reverse_encoder_dims

        # create the encoder
        encoder_layers = []
        for i in range(len(hidden_dims)):
            # each encoder layer goes from prev filter -> hidden_dims[i]
            # for each LSTM step. Then, we need to reconvolve onto the next
            # LSTM step, so we convolve to hidden_dims[i+1] filters and
            # max pool to downsample
            if i == 0:
                encoder_layers.append(ConvLSTM(prev_filt, hidden_dims[i], output_channels,
                                               (3, 3), (2, 2)))
            else:
                encoder_layers.append(ConvLSTM(prev_filt, hidden_dims[i], decoder_hidden_dims[::-1][i],
                                               (3, 3), (2, 2)))

            # update the filter value for the next iteration
            prev_filt = hidden_dims[i]

        self.encoder_layers = nn.ModuleList(encoder_layers)

        decoder_layers = []

        # the starting size is the last filter size of the encoder
        prev_decoder_filt = hidden_dims[-1]
        prev_encoder_filt = hidden_dims[-1]
        for i in range(len(decoder_hidden_dims)):
            # each decoder has the ith filter number of channels,
            # but (i+1)th filter in its cell state vector
            # in the UNet we also skip across the bottleneck, so the input is
            # the cat of both the skipped vector and the upsampling vector

            if i == 0:
                decoder_layers.append(ConvTransposeLSTM(prev_encoder_filt,
                                                        decoder_hidden_dims[i],
                                                        (3, 3), (2, 2)))
            else:
                decoder_layers.append(ConvTransposeLSTM(prev_decoder_filt + prev_encoder_filt,
                                                        decoder_hidden_dims[i],
                                                        (3, 3), (2, 2)))

            prev_decoder_filt = decoder_hidden_dims[i]
            prev_encoder_filt = reverse_encoder_dims[i]

        final_conv = nn.Conv3d(prev_decoder_filt, output_channels, (1, 3, 3), padding=(0, 1, 1), stride=(1, 1, 1))
        self.pred_final = nn.Sequential(Rearrange('b t c h w -> b c t h w'),
                                        final_conv,
                                        Rearrange('b c t h w -> b t c h w'))

        self.decoder_layers = nn.ModuleList(decoder_layers)

    def forward(self, x):
        '''
            Do the encode/decode loop
        '''
        xencs = []
        cencs = []
        c = None
        for i, layer in enumerate(self.encoder_layers):
            x, c = layer(x, None)
            cencs.append(c)
            xencs.append(x)

        xencs = xencs[::-1]
        cencs = cencs[::-1]

        for i, layer in enumerate(self.decoder_layers):
            if i == 0:
                x, c = layer(x, c=c)
            else:
                # skip the x vector across the bottleneck
                xconc = torch.cat([x, xencs[i]], dim=2)
                x, c = layer(xconc, c=cencs[i])

        # smooth the final output mask to remove the gridding
        x = self.pred_final(x)

        return x
