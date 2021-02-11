"""Bboxes encoder simple test."""
from absl import logging
import tensorflow as tf

import pytest

from detectia.config import Config
from detectia.engine.backbone import backbone_factory

# set tensorflow random seed.
tf.random.set_seed(111111)


@pytest.mark.parametrize(['model'], [
    pytest.param('efficientnet-b0'),
    pytest.param('efficientnet-b1'),
    pytest.param('efficientnet-b2'),
])
def test_sanity_efficientnet(model):
    backbone = backbone_factory.get_model(model)
    intermediate_features = backbone(tf.random.uniform(shape=(1, 416, 416, 3)))
    assert len(intermediate_features) == 6


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.test.main()
