
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


from collections import OrderedDict


# Initialize the weights of the network
def init_weights(m):
    if type(m) in [nn.Conv2d, nn.Conv1d]:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], np.prod(x.shape[1:]))



# Define our generic CNN model
class CNNModel(nn.Module):
    def __init__(
        self,
        win_size,
        inplanes=64,
        drop_prob=0.50,
        batch_norm=True,
        xavier=True,
        lrelu=True,
        conv_layer_chanls=None,
        bn_loc="bn_bf_relu",
        regression_label=False,
    ):

        # Define the number of layers, based on the window size
        number_of_layers_dict = {5: 2, 20: 3, 60: 4}
        number_of_layers = number_of_layers_dict[win_size]
        self.number_of_layers = number_of_layers

        # Define the empirical baseline settings, based on the window size
        EMP_CNN_BL_SETTING = {
            5: ([(5, 3)] * 10, 
                [(1, 1)] * 10, 
                [(1, 1)] * 10, 
                [(2, 1)] * 10),
            20: (
                [(5, 3)] * 10,
                [(3, 1)] + [(1, 1)] * 10,
                [(2, 1)] + [(1, 1)] * 10,
                [(2, 1)] * 10,
            ),
            60: (
                [(5, 3)] * 10,
                [(3, 1)] + [(1, 1)] * 10,
                [(3, 1)] + [(1, 1)] * 10,
                [(2, 1)] * 10,
            )
        }

        (filter_size_list,
        stride_list,
        dilation_list,
        max_pooling_list) = EMP_CNN_BL_SETTING[win_size]

        self.filter_size_list = filter_size_list
        self.stride_list = stride_list
        self.dilation_list = dilation_list
        self.max_pooling_list = max_pooling_list

        # Define the padding list based on the filter size list
        padding_list = [(int(fs[0] / 2), int(fs[1] / 2)) for fs in self.filter_size_list]
        self.padding_list = padding_list

        # Define the input size based on the window size
        input_size_dict = {5: (32, 15), 20: (64, 60), 60: (96, 180)}

        self.input_size = input_size_dict[win_size]
        self.conv_layer_chanls = conv_layer_chanls
        super(CNNModel, self).__init__()
        self.conv_layers = self._init_conv_layers(
            number_of_layers,
            inplanes,
            drop_prob,
            filter_size_list,
            stride_list,
            padding_list,
            dilation_list,
            max_pooling_list,
            batch_norm,
            lrelu,
            bn_loc,
        )
        fc_size = self._get_conv_layers_flatten_size()
        if regression_label:
            self.fc = nn.Linear(fc_size, 1)
        else:
            self.fc = nn.Linear(fc_size, 2)
        if xavier:
            self.conv_layers.apply(init_weights)
            self.fc.apply(init_weights)

    @staticmethod
    def conv_layer(
        in_chanl: int,
        out_chanl: int,
        lrelu=True,
        double_conv=False,
        batch_norm=True,
        bn_loc="bn_bf_relu",
        filter_size=(3, 3),
        stride=(1, 1),
        padding=1,
        dilation=1,
        max_pooling=(2, 2),
    ):
        assert bn_loc in ["bn_bf_relu", "bn_af_relu", "bn_af_mp"]

        if bn_loc == "bn_bf_relu":
            conv = [
                nn.Conv2d(
                    in_chanl,
                    out_chanl,
                    filter_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.BatchNorm2d(out_chanl),
                nn.LeakyReLU() if lrelu else nn.ReLU(),
            ]
        elif bn_loc == "bn_af_relu":
            conv = [
                nn.Conv2d(
                    in_chanl,
                    out_chanl,
                    filter_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.LeakyReLU() if lrelu else nn.ReLU(),
                nn.BatchNorm2d(out_chanl),
            ]
        else:
            conv = [
                nn.Conv2d(
                    in_chanl,
                    out_chanl,
                    filter_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                ),
                nn.LeakyReLU() if lrelu else nn.ReLU(),
            ]

        layers = conv * 2 if double_conv else conv

        if max_pooling != (1, 1):
            layers.append(nn.MaxPool2d(max_pooling, ceil_mode=True))

        if batch_norm and bn_loc == "bn_af_mp":
            layers.append(nn.BatchNorm2d(out_chanl))

        return nn.Sequential(*layers)

    def _init_conv_layers(
        self,
        layer_number,
        inplanes,
        drop_prob,
        filter_size_list,
        stride_list,
        padding_list,
        dilation_list,
        max_pooling_list,
        batch_norm,
        lrelu,
        bn_loc,
    ):
        if self.conv_layer_chanls is None:
            conv_layer_chanls = [inplanes * (2**i) for i in range(layer_number)]
        else:
            assert len(self.conv_layer_chanls) == layer_number
            conv_layer_chanls = self.conv_layer_chanls
        layers = []
        prev_chanl = 1
        for i, conv_chanl in enumerate(conv_layer_chanls):
            layers.append(
                self.conv_layer(
                    prev_chanl,
                    conv_chanl,
                    filter_size=filter_size_list[i],
                    stride=stride_list[i],
                    padding=padding_list[i],
                    dilation=dilation_list[i],
                    max_pooling=max_pooling_list[i],
                    batch_norm=batch_norm,
                    lrelu=lrelu,
                    bn_loc=bn_loc,
                )
            )
            prev_chanl = conv_chanl
        layers.append(Flatten())
        layers.append(nn.Dropout(p=drop_prob))
        return nn.Sequential(*layers)

    def _get_conv_layers_flatten_size(self):
        dummy_input = torch.rand((1, 1, self.input_size[0], self.input_size[1]))
        x = self.conv_layers(dummy_input)
        return x.shape[1]

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x


















