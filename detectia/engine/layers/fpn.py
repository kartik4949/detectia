# Copyright 2021 Kartik Sharma. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
""" Feature Pyramind Network Keras Layer. """
import tensorflow as tf
from absl import logging


class SepConvFPNBlock(tf.keras.layers.Layer):
    def __init__(self, config, name='SPConvBlock'):
        super().__init__(name=name)
        self.config = config
        self.ac = tf.keras.layers.LeakyReLU()
        self.bn = tf.keras.layers.BatchNormalization()
        self.dsc = tf.keras.layers.SeparableConv2D(
            config.fpn_num_filters, (3, 3), padding='same')
        self.up = tf.keras.layers.UpSampling2D(2)

    def call(self, x, training):
        x = self.dsc(x)
        x = self.bn(x, training=training)
        x = self.ac(x)
        return self.up(x)


class FPNBlock(tf.keras.layers.Layer):
    def __init__(self, config, name='fpn_block'):
        super().__init__(name=name)
        self.config = config
        self._fpn_block = []
        for i in range(self.config.num_fpn_nodes):
            logging.info('fpn_cell %d', i)
            self._fpn_block.append([SepConvFPNBlock(
                config, name=f'{name}_{i}/SPConvBlock/0'), SepConvFPNBlock(config, name=f'{name}_{i}/SPConvBlock/1')])

    def call(self, feats):
        _out_feats = []
        _skipped_feats = []
        _fpn_counter = 0
        for i, feat in enumerate(reversed(feats)):
            if i % 2 == 0:
                fpn_layer = self._fpn_block[_fpn_counter]
                if i == 0:
                    x = fpn_layer[0](feat)
                    _out_feats.append(x)
                    x = fpn_layer[1](x)
                else:
                    x = tf.concat([feat, x], axis=-1)
                    x = fpn_layer[0](x)
                    _out_feats.append(x)
                    x = fpn_layer[1](x)
                _fpn_counter += 1
            else:
                _skipped_feats.append(feat)
        return _out_feats, _skipped_feats
