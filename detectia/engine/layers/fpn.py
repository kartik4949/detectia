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
import functools

import tensorflow as tf
from absl import logging


class SepConvUpFPNBlock(tf.keras.layers.Layer):
    def __init__(self, config, name='SPUpConvBlock', **kwargs):
        super().__init__(name=name)
        self.config = config
        self.ac = tf.keras.layers.LeakyReLU()
        self.bn = tf.keras.layers.BatchNormalization()
        self.dsc = tf.keras.layers.SeparableConv2D(
            config.fpn_num_filters, (3, 3), padding='same',
            pointwise_initializer=tf.initializers.variance_scaling(),
            depthwise_initializer=tf.initializers.variance_scaling(), **kwargs)
        self.up = tf.keras.layers.UpSampling2D(2)

    def call(self, x, training):
        x = self.dsc(x)
        x = self.bn(x, training=training)
        x = self.ac(x)
        return self.up(x)


class SepConvFPNBlock(tf.keras.layers.Layer):
    def __init__(self, config, name='SPConvBlock', head=False, **kwargs):
        super().__init__(name=name)
        self.config = config
        self.ac = tf.keras.layers.LeakyReLU()
        self.bn = tf.keras.layers.BatchNormalization()
        if head:
            self.dsc = tf.keras.layers.Conv2D(
                config.num_anchors * (config.num_classes + 5), (1, 1), use_bias=False)
        else:
            self.dsc = tf.keras.layers.SeparableConv2D(
                config.fpn_num_filters,
                pointwise_initializer=tf.initializers.variance_scaling(),
                depthwise_initializer=tf.initializers.variance_scaling(), **kwargs)

    def call(self, x, training):
        x = self.dsc(x)
        x = self.bn(x, training=training)
        x = self.ac(x)
        return x


class FPNBlock(tf.keras.layers.Layer):
    def __init__(self, config, name='fpn_block'):
        super().__init__(name=name)
        self.config = config
        self._fpn_block = []
        for i in range(self.config.num_fpn_nodes):
            logging.info('fpn_cell %d', i)
            self._fpn_block.append([SepConvUpFPNBlock(
                config, name=f'{name}_{i}/SPUpConvBlock/0'), SepConvUpFPNBlock(config, name=f'{name}_{i}/SPUpConvBlock/1')])

    def call(self, feats):
        _out_feats = []
        for i in range(0, self.config.num_scales):
            feat_left = feats[-(i + 1)]
            feat_right = feats[-(i + self.config.num_scales)]
            if i == 0:
                x = self._fpn_block[i][0](feat_left)
                _out_feats.append(x)
                x = self._fpn_block[i][1](x)
            else:
                feat_left = self._fpn_block[i][0](feat_left)
                _out_feats.append(feat_left)
                x = tf.concat([feat_left, x], axis=-1)
                x = self._fpn_block[i][1](x)
            x = tf.concat([feat_right, x], axis=-1)
        _out_feats.append(x)
        return _out_feats


class FPNSequeeze(tf.keras.layers.Layer):
    def __init__(self, config, name='FPNSequeeze'):
        self.config = config
        super().__init__(name=name)
        self.fuse_ops = []
        for i in range(self.config.num_scales + 1):
            blocks = None
            _head_op = None
            if i == 0:
                _s2_squeeze = SepConvFPNBlock(
                    config=config, name='SequeezeFPNBlock_0_0', kernel_size=(2, 2), strides=(2, 2))
            else:
                _s2_squeeze = SepConvFPNBlock(
                    config=config, name=f'SequeezeFPNBlock_{i}_s2', kernel_size=(2, 2), strides=(2, 2))
                blocks = (SepConvFPNBlock(
                    config, name=f'SequeezeFPNBlock_{i}_{j}', padding='same', kernel_size=(3, 3)) for j in range((config.fpn_head_convs + i)))
                _head_op = SepConvFPNBlock(config=config,
                                           name=f'detection_head_{i}', head=True)
            self.fuse_ops.append([_s2_squeeze, blocks, _head_op])

    def call(self, out_feats, training):
        out_heads = []
        for i in range(len(out_feats)):
            out_feat = out_feats[-(i + 1)]
            if i == 0:
                x = self.fuse_ops[i][0](out_feat)
            else:
                x = tf.concat([x, out_feat], axis=-1)
                x = self.fuse_ops[i][0](x)
                for _fuse_op in self.fuse_ops[i][1]:
                    x = _fuse_op(x)
                x = self.fuse_ops[i][-1](x)
                out_heads.append(x)
        return out_heads
