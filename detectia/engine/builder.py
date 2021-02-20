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
""" Model Builder Class. """
import tensorflow as tf

from .backbone import backbone_factory
from .layers import fpn


class ModelBuilder(tf.keras.Model):
    """ Model Builder. """

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self._name = 'DetectiaNet'
        self.backbone = backbone_factory.get_model(config.backbone)
        self.fpn = fpn.FPNBuilder(config)

    @property
    def name(self):
        return self._name

    def call(self, input_image, training=False):
        """ DetectiaNet Call Method. """
        feats = self.backbone(input_image)
        detection_head_outs = self.fpn(feats)
        return detection_head_outs
