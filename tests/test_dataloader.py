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
"""test dataloader."""
import tensorflow as tf
from absl import logging

from detectia.config import Config
from detectia.datamodule import CreateRecords, DataLoader


class TestBoxEncode(tf.test.TestCase):
    def setUp(self):
        super().setUp()
        self.config = Config()
        self.temp_dir = self.get_temp_dir()
        self.tfrecord = CreateRecords()

    def test_sanity_dataloader(self):
        _dataloader = DataLoader(
            self.tfrecord.make_fake_tfrecord(self.temp_dir), self.config
        )
        dataset = _dataloader.from_tfrecords()
        data = next(iter(dataset))
        self.assertEqual(len(data), 5)


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.test.main()
