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
"""Bboxes encoder simple test."""
import tensorflow as tf
from absl import logging

from detectia.config import Config
from detectia.engine.box import BoxEncoder


class TestBoxEncode(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.config = Config()
        self.boxencoder = BoxEncoder(self.config)

    def test_sanity_encode_boxes(self):
        boxes = tf.constant(
            [[100, 120, 200, 202], [30, 50, 40, 80], [202, 400, 220, 410]], tf.float32
        )
        class_ids = tf.constant([1, 1, 0], tf.int32)
        targets = self.boxencoder.compute_targets(boxes, class_ids)
        self.assertEqual(len(targets), self.config.num_scales)


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.test.main()
