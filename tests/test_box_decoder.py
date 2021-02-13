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
"""Bboxes decoder simple test."""
import numpy as np
import tensorflow as tf
from absl import logging

from detectia.config import Config
from detectia.engine.box import BoxDecoder


class TestBoxEncode(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.config = Config()
        self.boxdecoder = BoxDecoder(self.config)

    def test_sanity_decode_boxes(self):
        model_output_feature = np.zeros(shape=(2, 13, 13, 3, 7))
        model_output_feature[0, 1, 10, 1] = [0.8, 0.8, 0.5, 0.5, 1.0, 0.0, 1.0]
        model_output_feature[1, 5, 5, 0] = [0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 1.0]
        feats = tf.constant(model_output_feature, tf.float32)
        anchors = [[10, 10], [20, 30], [50, 70]]
        decoded_outs = self.boxdecoder.decode_model_features(feats, anchors)
        self.assertEqual(len(decoded_outs), 4)


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.test.main()
