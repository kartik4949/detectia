"""test dataloader."""
from absl import logging
import tensorflow as tf

from detectia.datamodule import DataLoader
from detectia.utils import utils
from detectia.config import Config


class TestBoxEncode(tf.test.TestCase):

    def setUp(self):
        super().setUp()
        self.config = Config()
        self.temp_dir = self.get_temp_dir()

    def test_sanity_dataloader(self):
        _dataloader = DataLoader(utils.make_fake_tfrecord(self.temp_dir), self.config)
        dataset = _dataloader.from_tfrecords()
        data = next(iter(dataset))
        self.assertEqual(len(data), 7)


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.test.main()
