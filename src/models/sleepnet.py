# Shamelessly copied from OxWearables/asleep
from sklearn.model_selection import GroupShuffleSplit
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader
import torch.nn.init as init
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable

def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)

class Resnet(nn.Module):
    r""" The general form of the architecture can be described as follows:

    x->[Conv-[ResBlock]^m-BN-ReLU-Down]^n->y

    In other words:

            bn-relu-conv-bn-relu-conv
           /                         \
    x->conv --------------------------(+)-bn-relu-down->conv

    """

    def __init__(self, n_channels=3):
        super(Resnet, self).__init__()

        # Architecture definition. Each tuple defines
        # a basic Resnet layer Conv-[ResBlock]^m]-BN-ReLU-Down
        # isEva: change the classifier to two FC with ReLu
        # For example, (64, 5, 1, 5, 3, 1) means:
        # - 64 convolution filters
        # - kernel size of 5
        # - 1 residual block (ResBlock)
        # - ResBlock's kernel size of 5
        # - downsampling factor of 3
        # - downsampling filter order of 1
        # In the below, note that 3*3*5*5*4 = 900 (input size)
        cgf = [
            (64, 5, 2, 5, 3, 1),
            (128, 5, 2, 5, 3, 1),
            (256, 5, 2, 5, 5, 1),
            (512, 5, 2, 5, 5, 1),
            (1024, 5, 0, 5, 4, 0),
        ]  # 30 sec

        in_channels = n_channels
        feature_extractor = nn.Sequential()
        for i, layer_params in enumerate(cgf):
            (
                out_channels,
                conv_kernel_size,
                n_resblocks,
                resblock_kernel_size,
                downfactor,
                downorder,
            ) = layer_params
            feature_extractor.add_module(
                f"layer{i + 1}",
                Resnet.make_layer(
                    in_channels,
                    out_channels,
                    conv_kernel_size,
                    n_resblocks,
                    resblock_kernel_size,
                    downfactor,
                    downorder,
                ),
            )
            in_channels = out_channels

        self.feature_extractor = feature_extractor

    @staticmethod
    def make_layer(
        in_channels,
        out_channels,
        conv_kernel_size,
        n_resblocks,
        resblock_kernel_size,
        downfactor,
        downorder=1,
    ):
        r""" Basic layer in Resnets:

        x->[Conv-[ResBlock]^m-BN-ReLU-Down]->

        In other words:

                bn-relu-conv-bn-relu-conv
               /                         \
        x->conv --------------------------(+)-bn-relu-down->

        """

        # Check kernel sizes make sense (only odd numbers are supported)
        assert conv_kernel_size % 2, "Only odd number for conv_kernel_size supported"
        assert (
            resblock_kernel_size % 2
        ), "Only odd number for resblock_kernel_size supported"

        # Figure out correct paddings
        conv_padding = int((conv_kernel_size - 1) / 2)
        resblock_padding = int((resblock_kernel_size - 1) / 2)

        modules = [
            nn.Conv1d(
                in_channels,
                out_channels,
                conv_kernel_size,
                1,
                conv_padding,
                bias=False,
                padding_mode="circular",
            )
        ]

        for i in range(n_resblocks):
            modules.append(
                ResBlock(
                    out_channels,
                    out_channels,
                    resblock_kernel_size,
                    1,
                    resblock_padding,
                )
            )

        modules.append(nn.BatchNorm1d(out_channels))
        modules.append(nn.ReLU(True))
        modules.append(Downsample(out_channels, downfactor, downorder))

        return nn.Sequential(*modules)

    def forward(self, x):
        feats = self.feature_extractor(x)

        return feats


class ResBlock(nn.Module):
    r""" Basic bulding block in Resnets:

       bn-relu-conv-bn-relu-conv
      /                         \
    x --------------------------(+)->

    """

    def __init__(self, in_channels, out_channels,
                 kernel_size=5, stride=1, padding=2):
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            padding_mode="circular",
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(x))
        x = self.conv1(x)
        x = self.relu(self.bn2(x))
        x = self.conv2(x)

        x = x + identity

        return x


class Downsample(nn.Module):
    r"""Downsampling layer that applies anti-aliasing filters.
    For example, order=0 corresponds to a box filter (or average downsampling
    -- this is the same as AvgPool in Pytorch), order=1 to a triangle filter
    (or linear downsampling), order=2 to cubic downsampling, and so on.
    See https://richzhang.github.io/antialiased-cnns/ for more details.
    """

    def __init__(self, channels=None, factor=2, order=1):
        super(Downsample, self).__init__()
        assert factor > 1, "Downsampling factor must be > 1"
        self.stride = factor
        self.channels = channels
        self.order = order

        # Figure out padding and check params make sense
        # The padding is given by order*(factor-1)/2
        # so order*(factor-1) must be divisible by 2
        total_padding = order * (factor - 1)
        assert total_padding % 2 == 0, (
            "Misspecified downsampling parameters."
            "Downsampling factor and order must be such that "
            "order*(factor-1) is divisible by 2"
        )
        self.padding = int(order * (factor - 1) / 2)

        box_kernel = np.ones(factor)
        kernel = np.ones(factor)
        for _ in range(order):
            kernel = np.convolve(kernel, box_kernel)
        kernel /= np.sum(kernel)
        kernel = torch.Tensor(kernel)
        self.register_buffer(
            "kernel", kernel[None, None, :].repeat((channels, 1, 1)))

    def forward(self, x):
        return F.conv1d(
            x,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=x.shape[1])

class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes=2,
        lstm_layer=3,
        lstm_nn_size=1024,
        model_device="cpu",
        dropout_p=0,
        lstm_input_size=1024,
        bidrectional=False,
        batch_size=10,
        layer_norm=False,
    ):
        super(CNNLSTM, self).__init__()
        if bidrectional:
            fc_feature_size = lstm_nn_size * 2
        else:
            fc_feature_size = lstm_nn_size
        self.fc_feature_size = fc_feature_size
        self.model_device = model_device
        self.lstm_layer = lstm_layer
        self.batch_size = batch_size
        self.lstm_nn_size = lstm_nn_size
        self.bidrectional = bidrectional
        self.layer_norm = layer_norm
        self.feature_extractor = Resnet()
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_nn_size,
            num_layers=lstm_layer,
            bidirectional=bidrectional,
        )

        self.classifier = nn.Sequential(
            nn.Linear(fc_feature_size, fc_feature_size),
            nn.ReLU(True),
            nn.Dropout(p=dropout_p),
            nn.Linear(fc_feature_size, fc_feature_size),
            nn.ReLU(True),
            nn.Dropout(p=dropout_p),
            nn.Linear(fc_feature_size, num_classes),
        )

    def init_hidden(self, batch_size):
        # the weights are of the form (nb_layers, batch_size, nb_lstm_units)
        init_lstm_layer = self.lstm_layer
        if self.bidrectional:
            init_lstm_layer = self.lstm_layer * 2
        hidden_a = torch.randn(
            init_lstm_layer,
            batch_size,
            self.lstm_nn_size,
            device=self.model_device)
        hidden_b = torch.randn(
            init_lstm_layer,
            batch_size,
            self.lstm_nn_size,
            device=self.model_device)

        hidden_a = Variable(hidden_a)
        hidden_b = Variable(hidden_b)
        return hidden_a, hidden_b

    def forward(self, x):
        # 1. feature extractor
        # (B, C, L) -> (B, D, S) -> (B, S, D)
        x = self.feature_extractor(x)
        x = x.transpose(1,2)

        # 2. Sequence modelling
        # (B, S, D) -> (B, S, D*2)
        x, (_, _) = self.lstm(x)
        
        # 3. MLP readout
        # (B, S, D*2) -> (B, S, K)
        x = self.classifier(x)

        return x

def sleepnet(
    pretrained=True,
    my_device="cpu",
    num_classes=2,
    lstm_nn_size=128,
    dropout_p=0.5,
    bi_lstm=True,
    lstm_layer=1,
    local_weight_path=""
):
    model = CNNLSTM(
        num_classes=num_classes,
        model_device=my_device,
        lstm_nn_size=lstm_nn_size,
        dropout_p=dropout_p,
        bidrectional=bi_lstm,
        lstm_layer=lstm_layer,
    )
    weight_init(model)

    if pretrained:
        if len(local_weight_path) > 0:
            print("Loading local weight from %s" % local_weight_path)
            state_dict = torch.load(local_weight_path,
                                    map_location=torch.device(my_device))
            model.load_state_dict(
                state_dict)
        else:
            checkpoint = 'https://github.com/OxWearables/asleep/' \
                         'releases/download/0.4.9/sleepnet_apr_16_2024.mdl'
            model.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    checkpoint,
                    progress=True,
                    map_location=torch.device(my_device)))
    model.to(my_device, dtype=torch.float)
    return model

if __name__ == '__main__':
    B = 15
    inp = torch.randn((B,3,900*10))
    model = sleepnet(
        pretrained=True,
        my_device="cpu",
        num_classes=5,
        lstm_nn_size=1024,
        dropout_p=0.0,
        bi_lstm=True,
        lstm_layer=2,
        local_weight_path="",
    )
    num_params = sum(
            [p.numel() for p in model.parameters() if p.requires_grad]
    )
    print(num_params)
        # print(param)
    # out = model(inp)
    # fe_out = model.feature_extractor(inp)
    # print(fe_out.shape)
    # lstm_out, _ = model.lstm(fe_out.transpose(1,2))
    # print(lstm_out.shape)
    # c_out = model.classifier(lstm_out)
    # print(c_out.shape)
    # # out = model(inp, torch.Tensor([10 for _ in range(B)]))
    # print(out.shape)