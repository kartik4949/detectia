""" Create TFrecords from json GTs """
import os
import io

from PIL import Image
from absl import logging
import tensorflow as tf


class CreateRecords:
    """ TFRecord Utility Class. """
    @staticmethod
    def int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    @staticmethod
    def int64_list_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def bytes_list_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def float_list_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def create_tf_example(self, group):
        """create_tf_example.
        create tf example buffer

        Args:
            group: group name
        """
        with tf.io.gfile.GFile(os.path.join(group.filename), "rb") as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)

        width, height = image.size

        filename = group.filename.encode("utf8")
        image_format = b"jpg"
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for _, row in group.bbox.iterrows():
            xmins.append(row["xmin"] / width)
            xmaxs.append(row["xmax"] / width)
            ymins.append(row["ymin"] / height)
            ymaxs.append(row["ymax"] / height)
            if (
                row["xmin"] / width > 1
                or row["ymin"] / height > 1
                or row["xmax"] / width > 1
                or row["ymax"] / height > 1
            ):
                logging.info(row)

            classes_text.append(row["class"].encode("utf8"))
            classes.append(1)
        tf_example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/height": self.int64_feature(height),
                    "image/width": self.int64_feature(width),
                    "image/filename": self.bytes_feature(filename),
                    "image/image_id": self.bytes_feature("0".encode("utf8")),
                    "image/encoded": self.bytes_feature(encoded_jpg),
                    "image/format": self.bytes_feature(image_format),
                    "image/bbox/xmin": self.float_list_feature(xmins),
                    "image/bbox/xmax": self.float_list_feature(xmaxs),
                    "image/bbox/ymin": self.float_list_feature(ymins),
                    "image/bbox/ymax": self.float_list_feature(ymaxs),
                    "image/class/text": self.bytes_list_feature(classes_text),
                    "image/class/label": self.int64_list_feature(classes),
                }
            )
        )
        return tf_example

    def make_fake_tfrecord(self, temp_dir):
        """Makes fake TFRecord to test input."""
        tfrecord_path = os.path.join(temp_dir, 'test.tfrecords')
        writer = tf.io.TFRecordWriter(tfrecord_path)
        encoded_jpg = tf.io.encode_jpeg(tf.ones([512, 512, 3], dtype=tf.uint8))
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/height':
                        self.int64_feature(512),
                    'image/width':
                        self.int64_feature(512),
                    'image/filename':
                        self.bytes_feature('test_file_name.jpg'.encode(
                            'utf8')),
                    'image/source_id':
                        self.bytes_feature('123456'.encode('utf8')),
                    'image/key/sha256':
                        self.bytes_feature('qwdqwfw12345'.encode('utf8')),
                    'image/encoded':
                        self.bytes_feature(encoded_jpg.numpy()),
                    'image/format':
                        self.bytes_feature('jpeg'.encode('utf8')),
                    'image/object/bbox/xmin':
                        self.float_list_feature([0.1]),
                    'image/object/bbox/xmax':
                        self.float_list_feature([0.1]),
                    'image/object/bbox/ymin':
                        self.float_list_feature([0.2]),
                    'image/object/bbox/ymax':
                        self.float_list_feature([0.2]),
                    'image/object/class/text':
                        self.bytes_list_feature(['test'.encode('utf8')]),
                    'image/object/class/label':
                        self.int64_list_feature([1]),
                    'image/object/difficult':
                        self.int64_list_feature([]),
                    'image/object/truncated':
                        self.int64_list_feature([]),
                    'image/object/view':
                        self.bytes_list_feature([]),
                }))
        writer.write(example.SerializeToString())
        return tfrecord_path
