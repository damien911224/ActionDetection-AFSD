import torch.nn as nn
import torch.nn.functional as F


class MaxPool3dSamePadding(nn.MaxPool3d):

    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        batch, channel, t, h, w = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = [pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b]
        # print x.size()
        # print pad
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class TransposedConv1d(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=3,
                 stride=2,
                 padding=1,
                 output_padding=1,
                 activation_fn=F.relu,
                 use_batch_norm=False,
                 use_bias=True):
        super(TransposedConv1d, self).__init__()

        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn

        self.transposed_conv1d = nn.ConvTranspose1d(in_channels,
                                                    output_channels,
                                                    kernel_shape,
                                                    stride,
                                                    padding=padding,
                                                    output_padding=output_padding,
                                                    bias=use_bias)
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def forward(self, x):
        x = self.transposed_conv1d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class TransposedConv3d(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(3, 3, 3),
                 stride=(2, 1, 1),
                 padding=(1, 1, 1),
                 output_padding=(1, 0, 0),
                 activation_fn=F.relu,
                 use_batch_norm=False,
                 use_bias=True):
        super(TransposedConv3d, self).__init__()

        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn

        self.transposed_conv3d = nn.ConvTranspose3d(in_channels,
                                                    output_channels,
                                                    kernel_shape,
                                                    stride,
                                                    padding=padding,
                                                    output_padding=output_padding,
                                                    bias=use_bias)
        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    def forward(self, x):
        x = self.transposed_conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class Unit3D(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding='spatial_valid',
                 activation_fn=F.relu,
                 use_batch_norm=False,
                 use_bias=False):

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.padding = padding

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                bias=self._use_bias)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        if self.padding == 'same':
            (batch, channel, t, h, w) = x.size()
            pad_t = self.compute_pad(0, t)
            pad_h = self.compute_pad(1, h)
            pad_w = self.compute_pad(2, w)

            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            pad_h_f = pad_h // 2
            pad_h_b = pad_h - pad_h_f
            pad_w_f = pad_w // 2
            pad_w_b = pad_w - pad_w_f

            pad = [pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b]
            x = F.pad(x, pad)

        if self.padding == 'spatial_valid':
            (batch, channel, t, h, w) = x.size()
            pad_t = self.compute_pad(0, t)
            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f

            pad = [0, 0, 0, 0, pad_t_f, pad_t_b]
            x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class Unit1D(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=1,
                 stride=1,
                 padding='same',
                 activation_fn=F.relu,
                 use_bias=True):
        super(Unit1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels,
                                output_channels,
                                kernel_shape,
                                stride,
                                padding=0,
                                bias=use_bias)
        self._activation_fn = activation_fn
        self._padding = padding
        self._stride = stride
        self._kernel_shape = kernel_shape

    def compute_pad(self, t):
        if t % self._stride == 0:
            return max(self._kernel_shape - self._stride, 0)
        else:
            return max(self._kernel_shape - (t % self._stride), 0)

    def forward(self, x):
        if self._padding == 'same':
            batch, channel, t = x.size()
            pad_t = self.compute_pad(t)
            pad_t_f = pad_t // 2
            pad_t_b = pad_t - pad_t_f
            x = F.pad(x, [pad_t_f, pad_t_b])
        x = self.conv1d(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class ScaleTime(nn.Module):
    def __init__(self, channels):
        super(ScaleTime, self).__init__()
        self.scale_conv1 = nn.Sequential((
            nn.Conv2d(channels,
                      channels,
                      kernel_size=(1, 1),
                      stride=(1, 1),
                      padding=(0, 0),
                      dilation=(1, 1),
                      bias=True),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True)))
        self.scale_conv2 = nn.Sequential((
            nn.Conv2d(channels,
                      channels,
                      kernel_size=(3, 1),
                      stride=(1, 1),
                      padding=(1, 0),
                      dilation=(1, 1),
                      bias=True),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True)))
        self.scale_conv3 = nn.Sequential((
            nn.Conv2d(channels,
                      channels,
                      kernel_size=(3, 1),
                      stride=(1, 1),
                      padding=(2, 0),
                      dilation=(2, 1),
                      bias=True),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True)))
        self.scale_conv4 = nn.Sequential((
            nn.Conv2d(channels,
                      channels,
                      kernel_size=(5, 1),
                      stride=(1, 1),
                      padding=(2, 0),
                      dilation=(1, 1),
                      bias=True),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True)))

        self.scale_selection = nn.Sequential((
            nn.Conv2d(channels,
                      channels,
                      kernel_size=(1, 1),
                      stride=(1, 1),
                      padding=(0, 0),
                      dilation=(1, 1),
                      bias=True),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels,
                      4,
                      kernel_size=(1, 1),
                      stride=(1, 1),
                      padding=(0, 0),
                      dilation=(1, 1),
                      bias=True)
        ))
        self.scale_final_conv = nn.Sequential((
            nn.Conv2d(channels,
                      channels,
                      kernel_size=(1, 1),
                      stride=(1, 1),
                      padding=(0, 0),
                      dilation=(1, 1),
                      bias=True),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True)))

        self.time_conv1 = nn.Sequential((
            nn.Conv2d(channels,
                      channels,
                      kernel_size=(1, 3),
                      stride=(1, 1),
                      padding=(0, 1),
                      dilation=(1, 1),
                      bias=True),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True)))
        self.time_conv2 = nn.Sequential((
            nn.Conv2d(channels,
                      channels,
                      kernel_size=(1, 3),
                      stride=(1, 1),
                      padding=(0, 2),
                      dilation=(1, 2),
                      bias=True),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True)))
        self.time_conv3 = nn.Sequential((
            nn.Conv2d(channels,
                      channels,
                      kernel_size=(1, 3),
                      stride=(1, 1),
                      padding=(0, 3),
                      dilation=(1, 3),
                      bias=True),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True)))
        self.time_conv4 = nn.Sequential((
            nn.Conv2d(channels,
                      channels,
                      kernel_size=(1, 3),
                      stride=(1, 1),
                      padding=(0, 4),
                      dilation=(1, 4),
                      bias=True),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True)))

        self.time_selection = nn.Sequential((
            nn.Conv2d(channels,
                      channels,
                      kernel_size=(1, 1),
                      stride=(1, 1),
                      padding=(0, 0),
                      dilation=(1, 1),
                      bias=True),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels,
                      4,
                      kernel_size=(1, 1),
                      stride=(1, 1),
                      padding=(0, 0),
                      dilation=(1, 1),
                      bias=True)
        ))
        self.time_final_conv = nn.Sequential((
            nn.Conv2d(channels,
                      channels,
                      kernel_size=(1, 1),
                      stride=(1, 1),
                      padding=(0, 0),
                      dilation=(1, 1),
                      bias=True),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True)))

        self.feed_forward = nn.Sequential((
            nn.Conv2d(channels,
                      channels,
                      kernel_size=(1, 1),
                      stride=(1, 1),
                      padding=(0, 0),
                      dilation=(1, 1),
                      bias=True),
            nn.GroupNorm(32, channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(channels,
                      channels,
                      kernel_size=(1, 1),
                      stride=(1, 1),
                      padding=(0, 0),
                      dilation=(1, 1),
                      bias=True)
        ))

        self.norm1 = nn.GroupNorm(32, channels)
        self.dropout1 = nn.Dropout(0.1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.dropout2 = nn.Dropout(0.1)
        self.norm3 = nn.GroupNorm(32, channels)
        self.dropout3 = nn.Dropout(0.1)

    def forward(self, x):
        scale_x1 = self.scale_conv1(x)
        scale_x2 = self.scale_conv1(x)
        scale_x3 = self.scale_conv1(x)
        scale_x4 = self.scale_conv1(x)

        pooled = F.avg_pool2d(x, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0))
        # N, 4, S, T
        pooled = self.scale_selection(pooled)
        # N, 1, 4, S, T
        pooled = F.softmax(pooled.unsqueeze(1), dim=2)
        # N, C, 4, S, T
        outputs = torch.stack((scale_x1, scale_x2, scale_x3, scale_x4), dim=2)
        # N, C, S, T
        outputs = torch.sum(pooled * outputs, dim=2).squeeze(2)
        outputs = self.scale_final_conv(outputs)
        outputs = self.dropout1(outputs)

        outputs = outputs + x
        x = self.norm1(outputs)

        time_x1 = self.time_conv1(x)
        time_x2 = self.time_conv1(x)
        time_x3 = self.time_conv1(x)
        time_x4 = self.time_conv1(x)

        pooled = F.avg_pool2d(x, kernel_size=(1, 9), stride=(1, 1), padding=(0, 4))
        # N, 4, S, T
        pooled = self.time_selection(pooled)
        # N, 1, 4, S, T
        pooled = F.softmax(pooled.unsqueeze(1), dim=2)
        # N, C, 4, S, T
        outputs = torch.stack((time_x1, time_x2, time_x3, time_x4), dim=2)
        # N, C, S, T
        outputs = torch.sum(pooled * outputs, dim=2).squeeze(2)
        outputs = self.time_final_conv(outputs)
        outputs = self.dropout2(outputs)

        outputs = outputs + x
        x = self.norm2(outputs)

        outputs = self.feed_forward(x)
        outputs = self.dropout3(outputs)
        outputs = outputs + x
        outputs = self.norm3(outputs)

        return outputs
