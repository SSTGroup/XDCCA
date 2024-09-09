import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.constraints import unit_norm

from XDCCA.algorithms.losses_metrics import compute_l2


class Encoder(tf.keras.Model):
    def __init__(self, config, view_ind, **kwargs):
        super(Encoder, self).__init__(name=f"Encoder_view_{view_ind}", **kwargs)
        self.config = config
        self.view_index = view_ind

        self.dense_layers = {
            str(i): layers.Dense(
                dim,
                activation=activ,
            )
            for i, (dim, activ) in enumerate(self.config)
        }

    def get_l2(self):
        return tf.math.reduce_sum(
            [
                compute_l2(layer.trainable_variables[0])
                for id, layer in self.dense_layers.items()
            ]
        )

    def call(self, inputs):
        x = inputs
        for i in range(len(self.dense_layers)):
            x = self.dense_layers[str(i)](x)

        return x


class ConvEncoder(tf.keras.Model):
    def __init__(self, config, view_ind, **kwargs):
        super(ConvEncoder, self).__init__(name=f"Encoder_view_{view_ind}", **kwargs)
        self.config = config
        self.view_index = view_ind

        self.conv_layers = list()
        for layer_conf in self.config:
            # Check whether input config is valid and supported
            assert layer_conf["l_type"] in ["conv", "maxpool"]

            if layer_conf["l_type"] == "conv":
                self.conv_layers.append(
                    layers.Conv2D(
                        filters=layer_conf["n_filters"],
                        kernel_size=layer_conf["k_size"],
                        strides=1,
                        padding="same",
                        activation=layer_conf["activation"],
                    )
                )
            elif layer_conf["l_type"] == "maxpool":
                self.conv_layers.append(
                    layers.MaxPool2D(
                        pool_size=layer_conf["pool_size"],
                        strides=None,
                        padding="valid",
                    )
                )
            else:
                raise NotImplementedError

    def get_l2(self):
        return tf.math.reduce_sum(
            [
                (
                    compute_l2(layer.trainable_variables[0])
                    if len(layer.trainable_variables) > 0
                    else 0
                )
                for layer in self.conv_layers
            ]
        )

    def call(self, inputs):
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)

        return x
