"""Bboxes encoder simple test."""
from absl import logging
import tensorflow as tf

from detectia.engine.box import BoxEncoder
from detectia.config import Config


class TestBoxEncode(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        tf.random.set_seed(111111)
        self.config = Config()
        self.boxencoder = BoxEncoder(self.config)

    def test_sanity_encode_boxes(self):
        boxes = tf.constant(
            [[100, 120, 200, 202], [30, 50, 40, 80], [202, 400, 220, 410]], tf.float32)
        class_ids = tf.constant([1, 1, 0], tf.int32)
        targets = self.boxencoder.compute_targets(boxes, class_ids)
        self.assertEqual(len(targets), self.config.num_scales)


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.test.main()
