"""GridMask Augmentation simple test."""
import pytest
from absl import logging
import tensorflow as tf

from detectia.augment import augmentations

tf.random.set_seed(11111)


@pytest.mark.parametrize(['shape', 'n_boxes'], [
    pytest.param((416, 416, 3), 4),
    pytest.param((416, 400, 3), 4),
    pytest.param((416, 400, 3), 0),
    pytest.param((32, 32, 3), 2),
])
def test_gridmask_images(shape, n_boxes):
    """Verify transformed image shape is valid and syntax check."""
    images = tf.random.uniform(
        shape=shape, minval=0, maxval=255, dtype=tf.float32)
    if n_boxes:
        bboxes = tf.random.uniform(
            shape=(n_boxes, 4), minval=1, maxval=511, dtype=tf.int32)
    else:
        bboxes = []
    transform_images, _ = augmentations.GridMask(image_shape=shape)(images, bboxes)
    assert images.shape[1] == transform_images.shape[1]


@pytest.mark.parametrize(['shape', 'n_boxes'], [
    pytest.param((4, 416, 416, 3), 4),
    pytest.param((4, 416, 400, 3), 4),
    pytest.param((4, 32, 32, 3), 2),
])
def test_mosaic_augmentation(shape, n_boxes):
    """Verify transformed image shape is valid and syntax check."""
    images = tf.random.uniform(
        shape=shape, minval=0, maxval=255, dtype=tf.float32)
    if n_boxes:
        bboxes = tf.random.uniform(
            shape=(4, n_boxes, 4), minval=1, maxval=511, dtype=tf.int32)
    else:
        bboxes = []
    transform_images, mosaic_boxes = augmentations.Mosaic()(images, bboxes)
    assert n_boxes == mosaic_boxes[0].shape[0]


if __name__ == "__main__":
    logging.set_verbosity(logging.WARNING)
    tf.compat.v1.disable_eager_execution()
    tf.test.main()
