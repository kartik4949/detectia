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
"""simple configuration test."""
import tensorflow as tf
from absl import logging

from detectia.config import Config


class ConfigTest(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.config = Config()

    def test_overide(self):
        _default_values = [self.config.num_scales, self.config.input_image_shape]
        self.config.override("configs/config.yaml")
        self.assertNotAllEqual(
            [self.config.num_scales, self.config.input_image_shape], _default_values
        )


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.test.main()
