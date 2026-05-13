import numpy as np

from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops.operations as P


class GlobalAvgPooling(nn.Cell):
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = P.ReduceMean(keep_dims=False)

    def construct(self, x):
        return self.mean(x, (2, 3))


class ShuffleV2Block(nn.Cell):
    def __init__(self, inp, oup, mid_channels, *, ksize, stride):
        super(ShuffleV2Block, self).__init__()
        self.stride = stride
        self.mid_channels = mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            nn.Conv2d(in_channels=inp, out_channels=mid_channels, kernel_size=1, stride=1,
                      pad_mode='pad', padding=0, has_bias=False),
            nn.BatchNorm2d(num_features=mid_channels, momentum=0.9),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=ksize, stride=stride,
                      pad_mode='pad', padding=pad, group=mid_channels, has_bias=False),
            nn.BatchNorm2d(num_features=mid_channels, momentum=0.9),
            nn.Conv2d(in_channels=mid_channels, out_channels=outputs, kernel_size=1, stride=1,
                      pad_mode='pad', padding=0, has_bias=False),
            nn.BatchNorm2d(num_features=outputs, momentum=0.9),
            nn.ReLU(),
        ]
        self.branch_main = nn.SequentialCell(branch_main)

        if stride == 2:
            branch_proj = [
                nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=ksize, stride=stride,
                          pad_mode='pad', padding=pad, group=inp, has_bias=False),
                nn.BatchNorm2d(num_features=inp, momentum=0.9),
                nn.Conv2d(in_channels=inp, out_channels=inp, kernel_size=1, stride=1,
                          pad_mode='pad', padding=0, has_bias=False),
                nn.BatchNorm2d(num_features=inp, momentum=0.9),
                nn.ReLU(),
            ]
            self.branch_proj = nn.SequentialCell(branch_proj)
        else:
            self.branch_proj = None

    def construct(self, old_x):
        if self.stride == 1:
            x_proj, x = self.channel_shuffle(old_x)
            return P.Concat(1)((x_proj, self.branch_main(x)))
        if self.stride == 2:
            x_proj = old_x
            x = old_x
            return P.Concat(1)((self.branch_proj(x_proj), self.branch_main(x)))
        return old_x

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = P.Shape()(x)
        x = P.Reshape()(x, (batchsize * num_channels // 2, 2, height * width,))
        x = P.Transpose()(x, (1, 0, 2,))
        x = P.Reshape()(x, (2, -1, num_channels // 2, height, width,))
        return x[0], x[1]


class ShuffleNetV2Backbone(nn.Cell):
    def __init__(self, model_size='1.0x'):
        super(ShuffleNetV2Backbone, self).__init__()

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if model_size == '0.5x':
            self.stage_out_channels = [-1, 24, 48, 96, 192, 1024]
        elif model_size == '1.0x':
            self.stage_out_channels = [-1, 24, 116, 232, 464, 1024]
        elif model_size == '1.5x':
            self.stage_out_channels = [-1, 24, 176, 352, 704, 1024]
        elif model_size == '2.0x':
            self.stage_out_channels = [-1, 24, 244, 488, 976, 2048]
        else:
            raise ValueError(f"Unsupported model_size: {model_size}")

        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.SequentialCell([
            nn.Conv2d(in_channels=3, out_channels=input_channel, kernel_size=3, stride=2,
                      pad_mode='pad', padding=1, has_bias=False),
            nn.BatchNorm2d(num_features=input_channel, momentum=0.9),
            nn.ReLU(),
        ])

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')

        features = []
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            for i in range(numrepeat):
                if i == 0:
                    features.append(
                        ShuffleV2Block(input_channel, output_channel,
                                       mid_channels=output_channel // 2, ksize=3, stride=2)
                    )
                else:
                    features.append(
                        ShuffleV2Block(input_channel // 2, output_channel,
                                       mid_channels=output_channel // 2, ksize=3, stride=1)
                    )
                input_channel = output_channel

        self.features = nn.SequentialCell([*features])

        self.conv_last = nn.SequentialCell([
            nn.Conv2d(in_channels=input_channel, out_channels=self.stage_out_channels[-1], kernel_size=1, stride=1,
                      pad_mode='pad', padding=0, has_bias=False),
            nn.BatchNorm2d(num_features=self.stage_out_channels[-1], momentum=0.9),
            nn.ReLU()
        ])

        self.globalpool = GlobalAvgPooling()
        self.out_channels = self.stage_out_channels[-1]
        self._initialize_weights()

    def construct(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        x = self.features(x)
        x = self.conv_last(x)
        x = self.globalpool(x)
        return x

    def _initialize_weights(self):
        for name, m in self.cells_and_names():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    m.weight.set_parameter_data(
                        Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype('float32'))
                    )
                else:
                    m.weight.set_parameter_data(
                        Tensor(np.random.normal(0, 1.0 / m.weight.data.shape[1], m.weight.data.shape).astype('float32'))
                    )


class ShuffleNetV2Head(nn.Cell):
    def __init__(self, input_channel, num_classes, has_dropout=False):
        super(ShuffleNetV2Head, self).__init__()
        head = ([nn.Dense(input_channel, num_classes, has_bias=True)] if not has_dropout else
                [nn.Dropout(0.2), nn.Dense(input_channel, num_classes, has_bias=True)])
        self.head = nn.SequentialCell(head)
        self._initialize_weights()

    def construct(self, x):
        return self.head(x)

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, nn.Dense):
                m.weight.set_data(
                    Tensor(np.random.normal(0, 0.01, m.weight.data.shape).astype('float32'))
                )
                if m.bias is not None:
                    m.bias.set_data(Tensor(np.zeros(m.bias.data.shape, dtype='float32')))

    @property
    def get_head(self):
        return self.head


class ShuffleNetV2Combine(nn.Cell):
    def __init__(self, backbone, head, activation='None'):
        super(ShuffleNetV2Combine, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.head = head
        self.need_activation = False
        if activation == 'Sigmoid':
            self.activation = P.Sigmoid()
            self.need_activation = True
        elif activation == 'Softmax':
            self.activation = P.Softmax()
            self.need_activation = True

    def construct(self, x):
        x = self.backbone(x)
        x = self.head(x)
        if self.need_activation:
            x = self.activation(x)
        return x


def shufflenet_v2(backbone, head, activation='None'):
    return ShuffleNetV2Combine(backbone, head, activation)
