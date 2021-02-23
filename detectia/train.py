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
""" Training Class for training models. """
import tensorflow as tf

import register
from .config import Config
from .datamodule import dataloader
from .engine import layers, builder, backbone
from .utils import utils


flags.DEFINE_string(
    'tpu',
    default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')
flags.DEFINE_string(
    'project',
    default=None,
    help='Project name for the Cloud TPU')
flags.DEFINE_string(
    'zone',
    default=None,
    help='GCE zone of the TPU.')
flags.DEFINE_enum('strategy', None, ['tpu', 'multi_gpu', 'default'],
                  'Training: gpus for multi-gpu, if None, use TF default.')

FLAGS = flags.FLAGS

RESOLVER = None
if FLAGS.tpu:
    RESOLVER = utils.init_tpu_resolver(FLAGS.tpu, FLAGS.zone, FLAGS.project)
else:
    logging.info('## Not Using TPU for Training ##')

_STRATEGY_SUPPORTED = {'multi_gpu': tf.distribute.MirroredStrategy(),
                       'default': tf.distribute.get_strategy(),
                       'tpu': tf.distribute.TPUStrategy(RESOLVER)}


class DistributedMixin:
    """Distributed Training Utilities Mixin Class."""

    def _init_model_with_strategy(self, config):
        with self.strategy.scope():
            self.model = self.model_builder(config)
        return

    def _distributed_dataloader_wrapper(self):
        dataset = self.strategy.experimental_distribute_dataset(self.dataset)
        return dataset

    def get_strategy(self, config):
        assert config.strategy in _STRATEGY_SUPPORTED.keys(
        ), f'{config.strategy} Strategy is not supported.'
        self.strategy = _STRATEGY_SUPPORTED[config.strategy]
        return

    @tf.function
    def distributed_train_step(dist_inputs):
        per_replica_losses = mirrored_strategy.run(self.train_step, args=(dist_inputs,))
        return self.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                    axis=None)
