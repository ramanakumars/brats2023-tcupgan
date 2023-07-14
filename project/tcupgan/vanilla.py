import torch.nn as nn
import torch


class TemporalUNet(nn.Module):
    gen_type = 'UNet'

    def __init__(self, hidden_dims=[8, 16, 32], bottleneck_dims=[16, 8],
                 input_channels=1, output_channels=4):

        super(TemporalUNet, self).__init__()

        self.hidden_dims = hidden_dims
        prev_filt = input_channels

        # create the encoder
        encoder_layers = []
        for i in range(len(hidden_dims) - 1):
            downsample2d = [nn.Conv3d(prev_filt, hidden_dims[i],
                                      kernel_size=(1, 3, 3), padding=(0, 1, 1)),
                            nn.ReLU(True), nn.BatchNorm3d(hidden_dims[i]),
                            nn.MaxPool3d((1, 2, 2))]

            encoder_layers.append(nn.Sequential(*downsample2d))

            # update the filter value for the next iteration
            prev_filt = hidden_dims[i]

        self.encoder_layers = nn.ModuleList(encoder_layers)

        prev_filt = hidden_dims[-1]

        bottleneck_layers_enc = []
        for j, filt in enumerate(bottleneck_dims):
            bottleneck_layers_enc.extend([
                nn.Conv3d(prev_filt, filt, (1, 1, 1), padding=0,
                          stride=1),
                nn.ReLU(True),  # nn.LeakyReLU(0.2),
                nn.BatchNorm3d(filt)])
            prev_filt = filt

        bottleneck_layers_dec = []
        decode_bottleneck = [*bottleneck_dims[::-1], hidden_dims[-1]]
        for j, filt in enumerate(decode_bottleneck):
            bottleneck_layers_dec.extend([
                nn.Conv3d(prev_filt, filt, (1, 1, 1), padding=0,
                          stride=1),
                nn.ReLU(True),  # nn.LeakyReLU(0.2),
                nn.BatchNorm3d(filt)])
            prev_filt = filt

        self.bottleneck_enc = nn.Sequential(*bottleneck_layers_enc)
        self.bottleneck_dec = nn.Sequential(*bottleneck_layers_dec)

        decoder_layers = []

        # invert the hidden layer filters for the decoder
        # also add the input channel at the end
        decoder_hidden_dims = [*hidden_dims[::-1][1:], output_channels]

        # the starting size is the last filter size of the encoder
        hidden_dim = hidden_dims[-1]
        for i in range(len(decoder_hidden_dims) - 1):
            # for upsampling the hidden vectors
            us3d = nn.Upsample(scale_factor=(1, 2, 2))
            conv3d = nn.Conv3d(hidden_dim * 2,
                               decoder_hidden_dims[i + 1], (1, 3, 3),
                               stride=1, padding=(0, 1, 1))
            act3d = nn.ReLU(True)
            bn3d = nn.BatchNorm3d(decoder_hidden_dims[i + 1])

            decoder_layers.append(nn.Sequential(us3d, conv3d, act3d, bn3d))

            hidden_dim = decoder_hidden_dims[i + 1]

        if output_channels > 1:
            self.pred_final = nn.Softmax(dim=2)
        else:
            self.pred_final = nn.Sigmoid()

        self.decoder_layers = nn.ModuleList(decoder_layers)

    def encode(self, x):
        '''
            Create the vector embedding
        '''
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)

        return x

    def decode(self, x, c):
        '''
            Decode from the vector embedding. Note that
            this needs both the cell state and the
            bottlenecked feature
        '''
        for i, layer in enumerate(self.decoder_layers):
            x = layer(x)

        return x

    def forward(self, x):
        '''
            Do the encode/decode loop
        '''

        x = torch.swapaxes(x, 1, 2)

        xencs = []
        for i, layer in enumerate(self.encoder_layers):
            x = layer(x)
            xencs.append(x)

        # bottleneck the c dimension
        enc_x = self.bottleneck_enc(x)

        # decode the encoded c vector for reconstruction
        dec_x = self.bottleneck_dec(enc_x)

        nlayers = len(self.decoder_layers)
        for i, layer in enumerate(self.decoder_layers):
            # skip the x vector across the bottleneck
            xconc = torch.cat([dec_x, xencs[nlayers - i - 1]], dim=1)

            dec_x = layer(xconc)

        # smooth the final output mask to remove the gridding
        x = self.pred_final(dec_x)

        x = torch.swapaxes(x, 1, 2)

        return x
