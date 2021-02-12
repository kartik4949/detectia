""" BoundingBox DataLoader Class """
import os
from typing import Optional, Text, Tuple

from absl import logging
import tensorflow as tf

from ..engine.box import BoxEncoder
from .base_data_loader import BaseDataLoader
from ..augment import augment
__all__ = ["DataLoader"]


class TFDecoderMixin:
    """Tensorflow decoder."""

    KEYS_TO_FEATURES = {
        "image/encoded": tf.io.FixedLenFeature((), tf.string),
        "image/source_id": tf.io.FixedLenFeature((), tf.string, ""),
        "image/height": tf.io.FixedLenFeature((), tf.int64, -1),
        "image/width": tf.io.FixedLenFeature((), tf.int64, -1),
        "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
        "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
        "image/object/class/label": tf.io.VarLenFeature(tf.int64),
        "image/object/area": tf.io.VarLenFeature(tf.float32),
        "image/object/is_crowd": tf.io.VarLenFeature(tf.int64),
    }

    def _decode_image(self, parsed_tensors):
        """Decodes the image"""
        image = tf.io.decode_image(parsed_tensors["image/encoded"], channels=3)
        image.set_shape([None, None, 3])
        return image

    def _decode_boxes(self, parsed_tensors):
        """Concat box coordinates in the format of [xmin, ymin, xmax, ymax]."""
        xmin = parsed_tensors["image/object/bbox/xmin"]
        xmax = parsed_tensors["image/object/bbox/xmax"]
        ymin = parsed_tensors["image/object/bbox/ymin"]
        ymax = parsed_tensors["image/object/bbox/ymax"]
        return tf.stack([xmin, ymin, xmax, ymax], axis=-1)

    def decode(self, serialized_example):
        """Decode the serialized example."""
        parsed_tensors = tf.io.parse_single_example(
            serialized_example, self.KEYS_TO_FEATURES
        )
        for k in parsed_tensors:
            if isinstance(parsed_tensors[k], tf.SparseTensor):
                if parsed_tensors[k].dtype == tf.string:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=""
                    )
                else:
                    parsed_tensors[k] = tf.sparse.to_dense(
                        parsed_tensors[k], default_value=0
                    )

        image = self._decode_image(parsed_tensors)
        boxes = self._decode_boxes(parsed_tensors)
        decode_image_shape = tf.logical_or(
            tf.equal(parsed_tensors["image/height"], -1),
            tf.equal(parsed_tensors["image/width"], -1),
        )
        image_shape = tf.cast(tf.shape(image), dtype=tf.int64)

        parsed_tensors["image/height"] = tf.where(
            decode_image_shape, image_shape[0], parsed_tensors["image/height"]
        )
        parsed_tensors["image/width"] = tf.where(
            decode_image_shape, image_shape[1], parsed_tensors["image/width"]
        )

        decoded_tensors = {
            "image": image,
            "height": parsed_tensors["image/height"],
            "width": parsed_tensors["image/width"],
            "groundtruth_classes": parsed_tensors["image/object/class/label"],
            "groundtruth_boxes": boxes,
        }
        return decoded_tensors


class DataLoader(BaseDataLoader, TFDecoderMixin):
    """DataLoader class.
    DataLoader Class for  dataset,This class will provide
    data iterable with images,bboxs or images,targets with required
    augmentations.
    """

    def __init__(
        self,
        data_path: Text,
        config: Optional[dict] = None,
        training: bool = True,
    ):
        """__init__.

        Args:
            data_path: Dataset Path ,this is required in proper structure
            please see readme file for more details on structuring.
            config: Config File for setting the required configuration of datapipeline.
            training:Traning mode on or not?
        """
        if not isinstance(data_path, str):
            msg = f"datapath should be str but pass {type(data_path)}."
            logging.error(msg)
            raise TypeError("Only str allowed")
        if not os.path.exists(data_path):
            msg = f"path doesnt exists"
            logging.error(msg)
            raise TypeError("Path doesnt exists")

        self._data_path = data_path
        self.config = config
        self._training = training
        self._drop_remainder = self.config.drop_remainder
        self.augmenter = augment.Augment(self.config)
        self._per_shard = self.config.shard
        self.max_instances_per_image = self.config.max_instances_per_image
        self.boxencoder = BoxEncoder(config)

    @property
    def classes(self):
        return self._classes

    def parser(self) -> tf.data.Dataset:
        """parser for reading images and bbox from tensor records."""
        dataset = tf.data.Dataset.list_files(
            self.data_path,
            shuffle=self._training,
        )
        if self._training:
            dataset = dataset.repeat()
        dataset = dataset.interleave(
            self._fetch_records, num_parallel_calls=self.AUTOTUNE
        )

        dataset = dataset.with_options(self.optimized_options)
        if self._training:
            dataset = dataset.shuffle(self._per_shard)
        return dataset

    def encoder(self, *args) -> Tuple:
        image_ids, images, boxes, class_ids = args
        targets = self.boxencoder.compute_targets(boxes, class_ids)
        return image_ids, images, boxes, class_ids, targets

    def decoder(self, value) -> Tuple:
        """helper decoder, a wrapper around tfrecorde decoder."""
        data = self.decode(value)
        # TODO: remove hardcoded value.
        image_id = 1.0
        image = data["image"]
        boxes = data["groundtruth_boxes"]
        classes = data["groundtruth_classes"]
        return (image_id, image, boxes, classes)

    def from_tfrecords(self) -> tf.data.Dataset:
        """tf_records.
        Returns a iterable tf.data dataset ,which is configured
        with the config file passed with require augmentations.
        """
        dataset = self.parser()

        def decode_rawdata(input_records): return self.decoder(
            input_records
        )  # pylint: enable=g-long-lambda
        dataset = dataset.map(decode_rawdata, num_parallel_calls=self.AUTOTUNE)
        dataset = dataset.prefetch(self.config.batch_size)

        if self._training:
            dataset = dataset.map(
                lambda image_id, image, bbox, classes: self.augmenter(
                    image, bbox, image_id, classes, return_image_label=False
                )
            )

        dataset = dataset.map(lambda *args: self.encoder(*args))

        # pad to fixed length.
        dataset = dataset.map(
            self.pad_to_fixed_len,
            num_parallel_calls=self.AUTOTUNE,
        )

        # make batches.
        dataset = dataset.batch(
            self.config.batch_size, drop_remainder=self._drop_remainder
        )
        dataset = self.pretraining(dataset)
        return dataset

    @property
    def data_path(self):
        return self._data_path
