"""Bboxes decoder simple test."""
from absl import logging

import numpy as np
import tensorflow as tf

from detectia.engine.box import BoxDecoder
from detectia.config import Config


class TestBoxEncode(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        tf.random.set_seed(111111)
        self.config = Config()
        self.boxdecoder = BoxDecoder(self.config)

    def test_sanity_decode_boxes(self):
        model_output_feature = np.zeros(shape=(2, 13, 13, 3, 7))
        model_output_feature[0, 1, 10, 1] = [0.8, 0.8, 0.5, 0.5, 1.0, 0.0, 1.0]
        model_output_feature[1, 5, 5, 0] = [0.5, 0.5, 0.5, 0.5, 1.0, 0.0, 1.0]
        feats = tf.constant(
            model_output_feature, tf.float32)
        anchors = [[10, 10], [20, 30], [50, 70]]
        decoded_outs = self.boxdecoder.decode_model_features(feats, anchors)
        self.assertEqual(len(decoded_outs), 4)


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.test.main()
