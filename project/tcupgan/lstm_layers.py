import torch.nn as nn
import torch


def apply_on_channel(x, layer):
    '''
        Applies layer on x after switching the channel
        and time axis. Reverts the axes when returning

            input             apply layer             output
        (b, t, ch, y, x) -> (b, ch, t, y, x) -> (b, t, ch, y, x)
    '''
    x = torch.swapaxes(layer(
        torch.swapaxes(x, 1, 2)), 1, 2)

    return x


def time_distribute(x, layer):
    '''
        Applies layer over the temporal axis (default dim 1)
    '''

    nt = x.shape[1]

    outputs = []
    for t in range(nt):
        outi = layer(x[:, t, :])
        outputs.append(outi)

    return torch.stack(outputs, axis=1)


'''
From https://github.com/ndrplz/ConvLSTM_pytorch
'''


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size,
                 bias=False, dropout=0):
        """
        Initialize ConvLSTM cell. This outputs the feature vector
        at the next timestep and the cell state learned upto this
        point.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        # the conv2d is going to from a concatenation of the
        # input tensor and the last hidden state
        # to a series of 4 tensors (i, f, o, g), which correspond to the
        # input, forget, cell and output gates
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, x, hidden_state):
        # get the image sizes
        nbatch, nt, _, height, width = x.size()

        # if we do not pass the encoded data from the previous cell, create
        # a vector of zeros for both the h and c
        if hidden_state is None:
            h, c = self.init_hidden(nbatch, (height, width))
        else:
            h, c = hidden_state

            if (h is None) and (c is None):
                h, c = self.init_hidden(nbatch, (height, width))
            if h is None:
                h, _ = self.init_hidden(nbatch, (height, width))

        # create a storage location for the output feature vector
        output_feature = torch.zeros(nbatch, nt, self.hidden_dim,
                                     height, width,
                                     device=self.conv.weight.device)

        # loop through the time axis and get the feature vector
        for t in range(nt):
            h, c = self.get_next(x[:, t, :, :, :], cur_state=[h, c])
            output_feature[:, t, :, :, :] = h

        return output_feature, c

    def get_next(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # combine along the channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        # convolve the combined input
        combined_conv = self.conv(combined)

        if self.dropout is not None:
            combined_conv = self.dropout(combined_conv)

        # split into the separate gates
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1)

        # activate the gates
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # predict the next timestep
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        '''
            Initalize zeros for the hidden states
            in case they are not passed in. This is actually
            called from the ConvLSTM object
        '''
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width,
                            device=self.conv.weight.device, requires_grad=True),
                torch.zeros(batch_size, self.hidden_dim, height, width,
                            device=self.conv.weight.device, requires_grad=True))


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 next_dim, kernel_size, pool_size):
        super(ConvLSTM, self).__init__()

        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lstmlayer = ConvLSTMCell(input_dim=self.input_dim,
                                      hidden_dim=self.hidden_dim,
                                      kernel_size=self.kernel_size)

        # 2d max pooling on the hidden layers
        downsample2d = [nn.Conv2d(self.hidden_dim, next_dim, kernel_size=1, bias=False),
                        nn.MaxPool2d(pool_size)]
        # downsample2d = [nn.BatchNorm2d(self.hidden_dim),
        #                nn.MaxPool2d(pool_size)]

        # 3d max pool for preserving time domain
        self.downsample3d = nn.MaxPool3d((1, *pool_size))

        self.downsample2d = nn.Sequential(*downsample2d)

    def forward(self, x, hidden_state=None):
        '''
            Runs the ConvLSTM layer. Expects input of the shape
            (batch_size, time, features, height, width), and outputs
            the temporal feature vector of size
            (batch_size, time, self.hidden_dim, height, width) as well
            as the last hidden state and the cell state, both of which are
            (batch_size, self.hidden_dim, height, width)
        '''

        output_feature, c = self.lstmlayer(x, hidden_state)

        # apply max pooling on the LSTM result
        # h = self.downsample2d(output_feature[:, -1, :, :])
        c = self.downsample2d(c)
        output_feature = apply_on_channel(output_feature, self.downsample3d)

        return output_feature, c  # , h, c


class ConvTransposeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 kernel_size, pool_size):
        super(ConvTransposeLSTM, self).__init__()

        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lstmlayer = ConvLSTMCell(input_dim=self.input_dim,
                                      hidden_dim=self.hidden_dim,
                                      kernel_size=self.kernel_size)

        # for upsampling the hidden vectors
        convt2d = nn.ConvTranspose2d(self.hidden_dim,
                                     self.hidden_dim,
                                     pool_size, stride=pool_size)
        bn2d = nn.InstanceNorm2d(self.hidden_dim)
        act2d = nn.LeakyReLU(0.2, True)

        # upsampling the feature vector
        convt3d = nn.ConvTranspose3d(self.hidden_dim,
                                     self.hidden_dim,
                                     (1, *pool_size),
                                     stride=(1, *pool_size))

        act3d = nn.LeakyReLU(0.2, True)
        bn3d = nn.InstanceNorm3d(self.hidden_dim)
        self.upsample3d = nn.Sequential(convt3d, bn3d, act3d)

        self.upsample2d = nn.Sequential(convt2d, bn2d, act2d)

    def forward(self, x, h=None, c=None):
        '''
            Runs the ConvLSTM layer. Expects input of the shape
            (batch_size, time, features, height, width), and outputs
            the temporal feature vector of size
            (batch_size, time, self.hidden_dim, height, width) as well
            as the last hidden state and the cell state, both of which are
            (batch_size, self.hidden_dim, height, width)
        '''

        hidden_state = [h, c]

        output_feature, c = self.lstmlayer(x, hidden_state)

        # apply max pooling on the LSTM result
        c = self.upsample2d(c)  # , self.upsample2d)
        output_feature = apply_on_channel(output_feature, self.upsample3d)

        return output_feature, c


class UpSampleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim,
                 kernel_size, pool_size):
        super(UpSampleLSTM, self).__init__()

        self.kernel_size = kernel_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.lstmlayer = ConvLSTMCell(input_dim=self.input_dim,
                                      hidden_dim=self.hidden_dim,
                                      kernel_size=self.kernel_size)
        # for upsampling the hidden vectors
        self.upsample2d = nn.Upsample(scale_factor=pool_size)

        # upsampling the feature vector
        self.upsample3d = nn.Upsample(scale_factor=(1, *pool_size))

    def forward(self, x, h=None, c=None):
        '''
            Runs the ConvLSTM layer. Expects input of the shape
            (batch_size, time, features, height, width), and outputs
            the temporal feature vector of size
            (batch_size, time, self.hidden_dim, height, width) as well
            as the last hidden state and the cell state, both of which are
            (batch_size, self.hidden_dim, height, width)
        '''

        hidden_state = [h, c]

        output_feature, c = self.lstmlayer(x, hidden_state)

        # apply max pooling on the LSTM result
        c = self.upsample2d(c)  # , self.upsample2d)
        output_feature = torch.swapaxes(self.upsample3d(torch.swapaxes(output_feature, 1, 2)), 1, 2)

        return output_feature, c
