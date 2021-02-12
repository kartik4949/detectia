"""Bbox utils simple test."""
from absl import logging
import tensorflow as tf

import pytest

from detectia.engine import utils


@pytest.mark.parametrize(['box1', 'box2', 'iou'], [
    pytest.param([100, 100, 200, 200], [150, 150, 200, 200], 0.25),
    pytest.param([100, 100, 200, 200], [100, 100, 200, 200], 1.0),
    pytest.param([100, 100, 200, 200], [201, 201, 210, 210], 0.0),
])
def test_iou_boxes(box1, box2, iou):
    """ simple test for iou between different boxes. """
    box_iou = utils.compute_iou_boxes(box1, box2)
    assert box_iou == iou


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.compat.v1.disable_eager_executing()
    tf.test.main()
