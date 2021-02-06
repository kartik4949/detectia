"""simple configuration test."""
from absl import logging

import tensorflow as tf

from detectia.config import Config


class ConfigTest(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        self.config = Config()

    def test_overide(self):
        _default_values = [self.config.level, self.config.input_image_shape]
        self.config.override('configs/config.yaml')
        self.assertNotAllEqual(
            [self.config.level, self.config.input_image_shape], _default_values)


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.test.main()
