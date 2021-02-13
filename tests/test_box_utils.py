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
"""Bbox utils simple test."""
import pytest
import tensorflow as tf
from absl import logging

from detectia.engine import utils


@pytest.mark.parametrize(
    ["box1", "box2", "iou"],
    [
        pytest.param([100, 100, 200, 200], [150, 150, 200, 200], 0.25),
        pytest.param([100, 100, 200, 200], [100, 100, 200, 200], 1.0),
        pytest.param([100, 100, 200, 200], [201, 201, 210, 210], 0.0),
    ],
)
def test_iou_boxes(box1, box2, iou):
    """ simple test for iou between different boxes. """
    box_iou = utils.compute_iou_boxes(box1, box2)
    assert box_iou == iou


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.compat.v1.disable_eager_executing()
    tf.test.main()
