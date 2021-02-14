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
"""FPN simple test."""
import pytest
import tensorflow as tf
from absl import logging

from detectia.engine.backbone import backbone_factory
from detectia.engine.layers import fpn
from detectia.config import Config

# set tensorflow random seed.
tf.random.set_seed(111111)


@pytest.mark.parametrize(
    ["model"],
    [
        pytest.param("efficientnet-b0"),
        pytest.param("efficientnet-b1"),
        pytest.param("efficientnet-b2"),
    ],
)
def test_sanity_fpnblock(model):
    config = Config()
    backbone = backbone_factory.get_model(model)
    intermediate_features = backbone(tf.random.uniform(shape=(1, 416, 416, 3)))
    fpnblock = fpn.FPNBlock(config)
    fpn_outputs = fpnblock(intermediate_features[1:])
    assert len(fpn_outputs[0]) == 3


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.test.main()
