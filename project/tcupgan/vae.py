import torch.nn as nn
import torch
from .lstm_layers import ConvLSTM, ConvTransposeLSTM, UpSampleLSTM


def sample(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std


class LSTMVAE_old(nn.Module):
    def __init__(self, hidden_dims=[8, 16, 32], bottleneck_dims=[16, 8],
                 input_channels=1, final_act='elu'):

        super(LSTMVAE, self).__init__()

        self.hidden_dims = hidden_dims
        prev_filt = input_channels

        # create the encoder
        encoder_layers = []
        for i in range(len(hidden_dims) - 1):
            # each encoder layer goes from prev filter -> hidden_dims[i]
            # for each LSTM step. Then, we need to reconvolve onto the next
            # LSTM step, so we convolve to hidden_dims[i+1] filters and
            # max pool to downsample
            encoder_layers.append(ConvLSTM(prev_filt, hidden_dims[i],
                                           hidden_dims[i + 1], (3, 3), (2, 2)))

            # update the filter value for the next iteration
            prev_filt = hidden_dims[i]

        self.encoder_layers = nn.ModuleList(encoder_layers)

        prev_filt = hidden_dims[-1]

        bottleneck_layers_enc = []
        for j, filt in enumerate(bottleneck_dims):
            bottleneck_layers_enc.extend([
                nn.Conv2d(prev_filt, filt, (1, 1), padding=0,
                          stride=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(filt)])
            prev_filt = filt

        self.enc_mu = nn.Conv2d(filt, filt, (1, 1), padding=0, stride=1)
        self.enc_sig = nn.Conv2d(filt, filt, (1, 1), padding=0, stride=1)

        bottleneck_layers_dec = []
        decode_bottleneck = [*bottleneck_dims[::-1], hidden_dims[-1]]
        for j, filt in enumerate(decode_bottleneck):
            bottleneck_layers_dec.extend([
                nn.Conv2d(prev_filt, filt, (1, 1), padding=0,
                          stride=1),
                nn.LeakyReLU(0.2),
                nn.BatchNorm2d(filt)])
            prev_filt = filt

        self.bottleneck_enc = nn.Sequential(*bottleneck_layers_enc)
        self.bottleneck_dec = nn.Sequential(*bottleneck_layers_dec)

        decoder_layers = []

        # invert the hidden layer filters for the decoder
        # also add the input channel at the end
        decoder_hidden_dims = [*hidden_dims[::-1][1:], input_channels]

        # the starting size is the last filter size of the encoder
        hidden_dim = hidden_dims[-1]
        for i in range(len(decoder_hidden_dims) - 1):
            # each decoder has the ith filter number of channels,
            # but (i+1)th filter in its cell state vector
            # in the UNet we also skip across the bottleneck, so the input is
            # the cat of both the skipped vector and the upsampling vector

            decoder_layers.append(ConvTransposeLSTM(decoder_hidden_dims[i],
                                                    hidden_dim,
                                                    decoder_hidden_dims[i + 1],
                                                    (3, 3), (2, 2)))
            hidden_dim = decoder_hidden_dims[i + 1]

        if final_act == 'elu':
            self.pred_final = nn.ELU()
        else:
            self.pred_final = nn.Sigmoid()

        self.decoder_layers = nn.ModuleList(decoder_layers)

    def encode(self, x):
        '''
            Create the vector embedding
        '''

        c = None
        for i, layer in enumerate(self.encoder_layers):
            if i == 0:
                hidden = None
            else:
                hidden = [None, c]

            x, c = layer(x, hidden)

            print(x.shape)

        return x, c

    def decode(self, x, c):
        '''
            Decode from the vector embedding. Note that
            this needs both the cell state and the
            bottlenecked feature
        '''
        for i, layer in enumerate(self.decoder_layers):
            x, c = layer(x, c=c)

        return x

    def forward(self, x):
        '''
            Do the encode/decode loop
        '''
        enc_x, enc_c = self.encode(x)

        '''
        # bottleneck the c dimension
        bottleneck_c = self.bottleneck_enc(enc_c)

        c_mu = self.enc_mu(bottleneck_c)
        c_sig = self.enc_sig(bottleneck_c)

        sampled_c = sample(c_mu, c_sig)

        # decode the encoded c vector for reconstruction
        dec_c = self.bottleneck_dec(sampled_c)

        img = self.decode(enc_x, dec_c)

        img = self.pred_final(img)
        '''

        return enc_x  # img, c_mu, c_sig


class LSTMVAE(nn.Module):
    gen_type = 'VAE'

    def __init__(self, hidden_dims=[8, 16, 32], bottleneck_dims=[16, 8],
                 input_channels=1, output_channels=4, input_size=(96, 96),
                 batchnorm=False, final_act='elu'):

        super(LSTMVAE, self).__init__()

        self.hidden_dims = hidden_dims

        # create the encoder
        conv_layers = []
        conv_layers.append(nn.Conv3d(input_channels, 16, (1, 3, 3), (1, 1, 1), padding=(0, 1, 1)))
        conv_layers.append(nn.LeakyReLU(0.2))

        conv_layers.append(nn.Conv3d(16, 32, (1, 3, 3), (1, 1, 1), padding=(0, 1, 1)))
        conv_layers.append(nn.LeakyReLU(0.2))

        self.enc_conv_layers = nn.Sequential(*conv_layers)

        prev_filt = 32
        # create the encoder
        encoder_layers = []
        for i in range(len(hidden_dims) - 1):
            # each encoder layer goes from prev filter -> hidden_dims[i]
            # for each LSTM step. Then, we need to reconvolve onto the next
            # LSTM step, so we convolve to hidden_dims[i+1] filters and
            # max pool to downsample
            encoder_layers.append(ConvLSTM(prev_filt, hidden_dims[i],
                                           hidden_dims[i + 1], (5, 5), (2, 2)))
            input_size = [int(size / 2) for size in input_size]

            # update the filter value for the next iteration
            prev_filt = hidden_dims[i]

        self.bottleneck_size = input_size

        self.encoder_layers = nn.ModuleList(encoder_layers)

        flattened_size = prev_filt = hidden_dims[-1] * input_size[0] * input_size[1]

        self.unpacked_filter_dim = hidden_dims[-1]

        bottleneck_layers_enc_x = []
        bottleneck_layers_enc_c = []
        for j, filt in enumerate(bottleneck_dims):
            bottleneck_layers_enc_x.extend([
                nn.Linear(prev_filt, filt),
                nn.LeakyReLU(0.2),
            ])  # nn.LeakyReLU(0.2),
            # nn.BatchNorm2d(filt)])
            bottleneck_layers_enc_c.extend([
                nn.Linear(prev_filt, filt),
                nn.LeakyReLU(0.2),
            ])  # nn.LeakyReLU(0.2),
            prev_filt = filt

            if batchnorm:
                bottleneck_layers_enc_x.append(nn.BatchNorm2d(filt))
                bottleneck_layers_enc_c.append(nn.BatchNorm1d(filt))

        self.bottleneck_enc_x = nn.Sequential(*bottleneck_layers_enc_x)
        self.bottleneck_enc_c = nn.Sequential(*bottleneck_layers_enc_c)
        self.enc_musig_x = nn.Linear(filt, 2 * filt)
        self.enc_musig_c = nn.Linear(filt, 2 * filt)

        bottleneck_layers_dec_x = []
        bottleneck_layers_dec_c = []
        decode_bottleneck = [*bottleneck_dims[::-1], flattened_size]
        for j, filt in enumerate(decode_bottleneck):
            bottleneck_layers_dec_x.extend([
                nn.Linear(prev_filt, filt),
                nn.LeakyReLU(0.2),
            ])  # nn.LeakyReLU(0.2),
            # nn.BatchNorm1d(filt)])
            bottleneck_layers_dec_c.extend([
                nn.Linear(prev_filt, filt),
                nn.LeakyReLU(0.2),
            ])  # nn.LeakyReLU(0.2),
            prev_filt = filt

        self.bottleneck_dec_x = nn.Sequential(*bottleneck_layers_dec_x)
        self.bottleneck_dec_c = nn.Sequential(*bottleneck_layers_dec_c)

        decoder_layers = []

        # invert the hidden layer filters for the decoder
        # also add the input channel at the end
        decoder_hidden_dims = [*hidden_dims[::-1][:-1]]

        # the starting size is the last filter size of the encoder
        hidden_dim = hidden_dims[-1]
        for i in range(len(decoder_hidden_dims) - 1):
            # each decoder has the ith filter number of channels,
            # but (i+1)th filter in its cell state vector
            # in the UNet we also skip across the bottleneck, so the input is
            # the cat of both the skipped vector and the upsampling vector

            decoder_layers.append(UpSampleLSTM(decoder_hidden_dims[i],
                                               hidden_dim,
                                               decoder_hidden_dims[i + 1],
                                               (5, 5), (2, 2)))
            # decoder_layers.append(UpSampleLSTM(decoder_hidden_dims[i] * 2,
            #                                        hidden_dim,
            #                                        decoder_hidden_dims[i + 1],
            #                                        (3, 3), (2, 2)))
            hidden_dim = decoder_hidden_dims[i + 1]

        decoder_conv = []
        decoder_conv.append(nn.ConvTranspose3d(hidden_dim, hidden_dims[0], (1, 4, 4), (1, 2, 2), padding=(0, 1, 1)))
        decoder_conv.append(nn.LeakyReLU(0.2))

        decoder_conv.append(nn.ConvTranspose3d(hidden_dims[0], output_channels, (1, 3, 3), (1, 1, 1), padding=(0, 1, 1)))

        if final_act == 'elu':
            decoder_conv.append(nn.ELU())
        else:
            decoder_conv.append(nn.Sigmoid())

        self.decoder_layers = nn.ModuleList(decoder_layers)
        self.decoder_conv = nn.Sequential(*decoder_conv)

    def encode(self, x):
        '''
            Create the vector embedding
        '''
        x = torch.swapaxes(self.enc_conv_layers(torch.swapaxes(x, 1, 2)), 1, 2)

        c = None
        for i, layer in enumerate(self.encoder_layers):
            if i == 0:
                hidden = None
            else:
                hidden = [None, c]

            x, c = layer(x, hidden)

        c = c.reshape((c.shape[0], -1))
        x = x.reshape((x.shape[0], x.shape[1], -1))

        bottleneck_x = self.bottleneck_enc_x(x)
        bottleneck_c = self.bottleneck_enc_c(c)

        enc_musig_c = self.enc_musig_c(bottleneck_c)
        enc_musig_x = self.enc_musig_x(bottleneck_x)

        _, _, nfilt = enc_musig_x.shape

        nfilt = int(nfilt / 2)

        c_mu = enc_musig_c[:, :nfilt]
        c_sig = enc_musig_c[:, nfilt:]

        x_mu = enc_musig_x[:, :, :nfilt]
        x_sig = enc_musig_x[:, :, nfilt:]

        enc_x = sample(x_mu, x_sig)
        enc_c = sample(c_mu, c_sig)

        return x_mu, x_sig, enc_x, c_mu, c_sig, enc_c

    def decode(self, x, c):
        '''
            Decode from the vector embedding. Note that
            this needs both the cell state and the
            bottlenecked feature
        '''

        x = self.bottleneck_dec_x(x).reshape((x.shape[0], x.shape[1],
                                              self.unpacked_filter_dim, *self.bottleneck_size))
        c = self.bottleneck_dec_c(c).reshape((x.shape[0], self.unpacked_filter_dim, *self.bottleneck_size))

        for i, layer in enumerate(self.decoder_layers):
            # skip the x vector across the bottleneck
            x, c = layer(x, c=c)

        # smooth the final output mask to remove the gridding
        x = torch.swapaxes(
            self.decoder_conv(
                torch.swapaxes(x, 1, 2)
            ), 1, 2)

        return x

    def forward(self, x):
        '''
            Do the encode/decode loop
        '''

        x_mu, x_sig, enc_x, c_mu, c_sig, enc_c = self.encode(x)

        x = self.decode(enc_x, enc_c)

        return x
