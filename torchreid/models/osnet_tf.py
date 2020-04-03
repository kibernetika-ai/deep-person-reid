from __future__ import division, absolute_import
import warnings
import torch
from torch import nn
from torch.nn import functional as F

import tensorflow as tf
from tensorflow.keras import layers


class ConvLayerTF(layers.Layer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding="SAME",
            groups=1,
    ):
        super(ConvLayerTF, self).__init__()
        self.conv = layers.Conv2D(
            out_channels,
            kernel_size,
            strides=stride,
            padding=padding,
            use_bias=False,
            # kernel_initializer=tf.keras.initializers.VarianceScaling(
            #     scale=2., mode="fan_out", distribution="truncated_normal"
            # ),
        )
        # self.bn = layers.BatchNormalization(axis=3, epsilon=1e-5, momentum=0.1)
        self.bn = layers.BatchNormalization(axis=3, epsilon=1e-5, momentum=0.1)
        self.relu = layers.ReLU()

    def call(self, inputs, **kwargs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1TF(layers.Layer):
    """1x1 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv1x1TF, self).__init__()
        self.conv = layers.Conv2D(
            out_channels,
            kernel_size=1,
            strides=stride,
            padding="SAME",
            use_bias=False,
            # kernel_initializer=tf.keras.initializers.VarianceScaling(
            #     scale=2., mode="fan_out", distribution="truncated_normal"
            # ),
        )
        # self.bn = layers.BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5)
        self.bn = layers.BatchNormalization(axis=3, epsilon=1e-5, momentum=0.1)
        self.relu = layers.ReLU()

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1x1LinearTF(layers.Layer):
    """1x1 convolution + bn (w/o non-linearity)."""

    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv1x1LinearTF, self).__init__()
        self.conv = layers.Conv2D(
            out_channels, 1, strides=stride, padding="SAME", use_bias=False,
            # kernel_initializer=tf.keras.initializers.VarianceScaling(
            #     scale=2., mode="fan_out", distribution="truncated_normal"
            # ),
        )
        # self.bn = layers.BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5)
        self.bn = layers.BatchNormalization(axis=3, epsilon=1e-5, momentum=0.1)

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.bn(x)
        return x


class Conv3x3TF(layers.Layer):
    """3x3 convolution + bn + relu."""

    def __init__(self, in_channels, out_channels, stride=1, groups=1):
        super(Conv3x3TF, self).__init__()
        self.conv = layers.Conv2D(
            out_channels,
            kernel_size=3,
            strides=stride,
            padding="SAME",
            use_bias=False,
            # kernel_initializer=tf.keras.initializers.VarianceScaling(
            #     scale=2., mode="fan_out", distribution="truncated_normal"
            # ),
        )
        # self.bn = layers.BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5)
        self.bn = layers.BatchNormalization(axis=3, epsilon=1e-5, momentum=0.1)
        self.relu = layers.ReLU()

    def call(self, x, **kwargs):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class LightConv3x3TF(layers.Layer):
    """Lightweight 3x3 convolution.

    1x1 (linear) + dw 3x3 (nonlinear).
    """

    def __init__(self, in_channels, out_channels):
        super(LightConv3x3TF, self).__init__()
        self.conv1 = layers.Conv2D(
            out_channels, kernel_size=1, strides=1, padding="SAME", use_bias=False
        )
        self.conv2 = layers.SeparableConv2D(
            out_channels,
            kernel_size=3,
            strides=1,
            padding="SAME",
            use_bias=False,
            depth_multiplier=out_channels,
        )
        # self.bn = layers.BatchNormalization(axis=3, momentum=0.1, epsilon=1e-5)
        self.bn = layers.BatchNormalization(axis=3, epsilon=1e-5, momentum=0.1)
        self.relu = layers.ReLU()

    def call(self, x, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ChannelGateTF(layers.Layer):
    """A mini-network that generates channel-wise gates conditioned on input tensor."""

    def __init__(
            self,
            in_channels,
            num_gates=None,
            return_gates=False,
            gate_activation='sigmoid',
            reduction=16,
    ):
        super(ChannelGateTF, self).__init__()
        if num_gates is None:
            num_gates = in_channels
        self.in_channels = in_channels
        self.return_gates = return_gates
        self.global_avgpool = layers.GlobalAvgPool2D()
        # self.global_avgpool = layers.AveragePooling2D(1)
        self.fc1 = layers.Conv2D(
            in_channels // reduction,
            kernel_size=1,
            use_bias=True,
            padding="SAME",
            # kernel_initializer=tf.keras.initializers.VarianceScaling(
            #     scale=2., mode="fan_out", distribution="truncated_normal"
            # ),
            bias_initializer=tf.keras.initializers.Zeros()
        )
        self.norm1 = None
        self.relu = layers.ReLU()
        self.fc2 = layers.Conv2D(
            num_gates,
            kernel_size=1,
            use_bias=True,
            padding="SAME",
            # kernel_initializer=tf.keras.initializers.VarianceScaling(
            #     scale=2., mode="fan_out", distribution="truncated_normal"
            # ),
            bias_initializer=tf.keras.initializers.Zeros()
        )
        if gate_activation == 'sigmoid':
            self.gate_activation = tf.keras.activations.sigmoid
        elif gate_activation == 'relu':
            self.gate_activation = layers.ReLU()
        elif gate_activation == 'linear':
            self.gate_activation = None
        else:
            raise RuntimeError(
                "Unknown gate activation: {}".format(gate_activation)
            )

    def call(self, x, **kwargs):
        input = x
        x = self.global_avgpool(x)
        x = layers.Reshape([1, 1, self.in_channels])(x)
        x = self.fc1(x)
        if self.norm1 is not None:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.fc2(x)
        if self.gate_activation is not None:
            x = self.gate_activation(x)
        if self.return_gates:
            return x
        return input * x


class OSBlockTF(layers.Layer):
    """Omni-scale feature learning block."""

    def __init__(
            self,
            in_channels,
            out_channels,
            bottleneck_reduction=4,
            **kwargs
    ):
        super(OSBlockTF, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        self.conv1 = Conv1x1TF(in_channels, mid_channels)
        self.conv2a = LightConv3x3TF(mid_channels, mid_channels)
        self.conv2b = tf.keras.Sequential([
            LightConv3x3TF(mid_channels, mid_channels),
            LightConv3x3TF(mid_channels, mid_channels),
        ])
        self.conv2c = tf.keras.Sequential([
            LightConv3x3TF(mid_channels, mid_channels),
            LightConv3x3TF(mid_channels, mid_channels),
            LightConv3x3TF(mid_channels, mid_channels),
        ])
        self.conv2d = tf.keras.Sequential([
            LightConv3x3TF(mid_channels, mid_channels),
            LightConv3x3TF(mid_channels, mid_channels),
            LightConv3x3TF(mid_channels, mid_channels),
            LightConv3x3TF(mid_channels, mid_channels),
        ])
        self.gate = ChannelGateTF(mid_channels)
        self.conv3 = Conv1x1LinearTF(mid_channels, out_channels)
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = Conv1x1LinearTF(in_channels, out_channels)
        self.IN = None

    def call(self, x, **kwargs):
        identity = x
        x1 = self.conv1(x)
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x1)
        x2c = self.conv2c(x1)
        x2d = self.conv2d(x1)
        x2 = self.gate(x2a) + self.gate(x2b) + self.gate(x2c) + self.gate(x2d)
        x3 = self.conv3(x2)
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = x3 + identity
        if self.IN is not None:
            out = self.IN(out)
        return tf.keras.activations.relu(out)


##########
# Network architecture
##########
class OSNetTF(tf.keras.Model):
    """Omni-Scale Network.

    Reference:
        - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        - Zhou et al. Learning Generalisable Omni-Scale Representations
          for Person Re-Identification. arXiv preprint, 2019.
    """

    def __init__(
            self,
            num_classes,
            blocks,
            m_layers,
            channels,
            feature_dim=512,
            loss='softmax',
            **kwargs
    ):
        super(OSNetTF, self).__init__()
        num_blocks = len(blocks)
        assert num_blocks == len(m_layers)
        assert num_blocks == len(channels) - 1
        self.loss = loss
        self.channels = channels

        # convolutional backbone
        self.conv1 = ConvLayerTF(3, channels[0], 7, stride=2, padding="SAME")  # original padding=3
        # self.maxpool = nn.MaxPool2d(3, stride=2, padding="SAME")  # original padding=1
        self.maxpool = layers.MaxPool2D(3, strides=2, padding="SAME")  # original padding=1
        self.conv2 = self._make_layer(
            blocks[0],
            m_layers[0],
            channels[0],
            channels[1],
            reduce_spatial_size=True,
        )
        self.conv3 = self._make_layer(
            blocks[1],
            m_layers[1],
            channels[1],
            channels[2],
            reduce_spatial_size=True
        )
        self.conv4 = self._make_layer(
            blocks[2],
            m_layers[2],
            channels[2],
            channels[3],
            reduce_spatial_size=False
        )
        self.conv5 = Conv1x1TF(channels[3], channels[3])
        self.global_avgpool = layers.GlobalAvgPool2D()
        # fully connected layer
        self.fc = self._construct_fc_layer(
            feature_dim, channels[3], dropout_p=None
        )
        # identity classification layer
        # self.classifier = nn.Linear(self.feature_dim, num_classes)
        # self.classifier = layers.Dense(num_classes)
        self.classifier = layers.Dense(num_classes, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01))

        # self._init_params()

    def _make_layer(
            self,
            block,
            layer,
            in_channels,
            out_channels,
            reduce_spatial_size,
    ):
        m_layers = [block(in_channels, out_channels)]

        for i in range(1, layer):
            m_layers.append(block(out_channels, out_channels))

        if reduce_spatial_size:
            m_layers.append(
                tf.keras.Sequential([
                    Conv1x1TF(out_channels, out_channels),
                    layers.AvgPool2D(2, strides=2)
                ])
            )

        return tf.keras.Sequential(m_layers)

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        if fc_dims is None or fc_dims < 0:
            self.feature_dim = input_dim
            return None

        if isinstance(fc_dims, int):
            fc_dims = [fc_dims]

        m_layers = []
        for dim in fc_dims:
            # m_layers.append(layers.Dense(dim))
            m_layers.append(layers.Dense(dim, kernel_initializer=tf.keras.initializers.RandomNormal(0, 0.01)))
            # 2 Dimensions: [batch, channels], need normalization on channels
            # m_layers.append(layers.BatchNormalization(axis=1, momentum=0.1, epsilon=1e-5))
            m_layers.append(layers.BatchNormalization(axis=1, momentum=0.1, epsilon=1e-5))
            m_layers.append(layers.ReLU())
            if dropout_p is not None:
                m_layers.append(layers.Dropout(rate=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return tf.keras.Sequential(m_layers)

    # def _init_params(self):
    #     for m in self.layers:
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(
    #                 m.weight, mode='fan_out', nonlinearity='relu'
    #             )
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #
    #         elif isinstance(m, nn.BatchNorm2d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #
    #         elif isinstance(m, nn.BatchNorm1d):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)
    #
    #         elif isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, 0, 0.01)
    #             if m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)

    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

    def call(self, x, **kwargs):
        return_featuremaps = kwargs.get('return_featuremaps')
        x = self.featuremaps(x)
        if return_featuremaps:
            return x
        v = self.global_avgpool(x)
        v = layers.Reshape([self.channels[-1]])(v)
        # v = tf.reshape(v, [v.shape[0], -1])
        # v = v.view(v.size(0), -1)
        if self.fc is not None:
            v = self.fc(v)
        if not self.trainable:
            return v
        y = self.classifier(v)
        # tf.print(tf.reduce_max(v))
        return y, v


##########
# Instantiation
##########
def osnet_x1_0(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    # standard size (width x1.0)
    model = OSNetTF(
        num_classes,
        blocks=[OSBlockTF, OSBlockTF, OSBlockTF],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        **kwargs
    )
    # if pretrained:
    #     init_pretrained_weights(model, key='osnet_x1_0')
    return model


def osnet_x1_0_fixed(num_classes=751, pretrained=False, loss='softmax', **kwargs):
    # standard size (width x1.0)
    model = OSNetTF(
        num_classes,
        blocks=[OSBlockTF, OSBlockTF, OSBlockTF],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        **kwargs
    )
    # if pretrained:
    #     init_pretrained_weights(model, key='osnet_x1_0')
    return model


def osnet_x0_75(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    # medium size (width x0.75)
    model = OSNetTF(
        num_classes,
        blocks=[OSBlockTF, OSBlockTF, OSBlockTF],
        layers=[2, 2, 2],
        channels=[48, 192, 288, 384],
        loss=loss,
        **kwargs
    )
    # if pretrained:
    #     init_pretrained_weights(model, key='osnet_x0_75')
    return model


def osnet_x0_5(num_classes=1000, pretrained=True, loss='softmax', **kwargs):
    # tiny size (width x0.5)
    model = OSNetTF(
        num_classes,
        blocks=[OSBlockTF, OSBlockTF, OSBlockTF],
        layers=[2, 2, 2],
        channels=[32, 128, 192, 256],
        loss=loss,
        **kwargs
    )
    # if pretrained:
    #     init_pretrained_weights(model, key='osnet_x0_5')
    return model


def osnet_x0_25(num_classes=751, loss='softmax', **kwargs):
    # very tiny size (width x0.25)
    model = OSNetTF(
        num_classes,
        blocks=[OSBlockTF, OSBlockTF, OSBlockTF],
        m_layers=[2, 2, 2],
        channels=[16, 64, 96, 128],
        loss=loss,
        **kwargs
    )
    # if pretrained:
    #     init_pretrained_weights(model, key='osnet_x0_25')
    return model


def osnet_ibn_x1_0(
        num_classes=1000, pretrained=True, loss='softmax', **kwargs
):
    # standard size (width x1.0) + IBN layer
    # Ref: Pan et al. Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net. ECCV, 2018.
    model = OSNetTF(
        num_classes,
        blocks=[OSBlockTF, OSBlockTF, OSBlockTF],
        layers=[2, 2, 2],
        channels=[64, 256, 384, 512],
        loss=loss,
        IN=True,
        **kwargs
    )
    # if pretrained:
    #     init_pretrained_weights(model, key='osnet_ibn_x1_0')
    return model
