import os

import tensorflow as tf


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def make_fake_tfrecord(temp_dir):
    """Makes fake TFRecord to test input."""
    tfrecord_path = os.path.join(temp_dir, 'test.tfrecords')
    writer = tf.io.TFRecordWriter(tfrecord_path)
    encoded_jpg = tf.io.encode_jpeg(tf.ones([512, 512, 3], dtype=tf.uint8))
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height':
                    int64_feature(512),
                'image/width':
                    int64_feature(512),
                'image/filename':
                    bytes_feature('test_file_name.jpg'.encode(
                        'utf8')),
                'image/source_id':
                    bytes_feature('123456'.encode('utf8')),
                'image/key/sha256':
                    bytes_feature('qwdqwfw12345'.encode('utf8')),
                'image/encoded':
                    bytes_feature(encoded_jpg.numpy()),
                'image/format':
                    bytes_feature('jpeg'.encode('utf8')),
                'image/object/bbox/xmin':
                    float_list_feature([0.1]),
                'image/object/bbox/xmax':
                    float_list_feature([0.1]),
                'image/object/bbox/ymin':
                    float_list_feature([0.2]),
                'image/object/bbox/ymax':
                    float_list_feature([0.2]),
                'image/object/class/text':
                    bytes_list_feature(['test'.encode('utf8')]),
                'image/object/class/label':
                    int64_list_feature([1]),
                'image/object/difficult':
                    int64_list_feature([]),
                'image/object/truncated':
                    int64_list_feature([]),
                'image/object/view':
                    bytes_list_feature([]),
            }))
    writer.write(example.SerializeToString())
    return tfrecord_path
