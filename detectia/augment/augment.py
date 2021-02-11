import functools
import inspect
from typing import List, Text, Dict

import tensorflow as tf
import tensorflow_addons as tfa

from ..augment import augmentations

ALLOWED_TRANSFORMATIONS = [
    "flip_left_right",
    "random_rotate",
    "gridmask",
    "random_rotate",
    "random_shear_y",
    "cutout",
    "mosaic",
    "random_shear_x",
]


"""Augment Class for interface of Augmentations."""


class Augmentation(augmentations.TransformMixin):
    """Augmentation.
    Class Augmentation which consists inhouse augmentations and builds
    the transformations pipeline with given transformations in config.
    ::

    Example:

        augment = Augmentation(config,["random_rotate","gridmask"])
        # Use the pipeline and iterate over function in the pipeline.
        pipeline = augment._pipeline

    """

    def __init__(self, config: Dict, transformations: Dict, type: Text = "bbox"):
        """__init__.
                Augmentation class provides and builds the augmentations pipe-
                line required for tf.data iterable.

        Args:
            config: config file containing augmentations and kwargs required.
            transformations: transformations contains list of augmentations
            to build the pipeline one.
            type: type of dataset to built the pipeline for e.g bbox,
            keypoints,categorical,etc.
        """
        self.config = config
        self.type = type
        self.transformations = transformations
        self._pipeline = []
        self._set_tfa_attrb()
        self.image_size = config.input_image_shape
        # builds the augment pipeline.
        self._pipeline.append(
            (functools.partial(tf.image.resize, size=self.image_size), False)
        )

        for transform, kwargs in transformations.items():
            if transform not in ALLOWED_TRANSFORMATIONS and not hasattr(
                tf.image, transform
            ):
                raise ValueError(
                    f"{transform} is not a valid augmentation for \
                            tf.image or TensorPipe,please visit readme section"
                )

            kwargs = kwargs if isinstance(kwargs, Dict) else {}

            if hasattr(tf.image, transform):
                transform = getattr(tf.image, transform)
                transform = functools.partial(transform, **kwargs)
                self._pipeline.append((transform, False))
            else:
                transform = getattr(self, transform)
                transform = functools.partial(transform, **kwargs)
                self._pipeline.append((transform, True))

    def _set_tfa_attrb(self):
        """_set_tfa_attrb.
        helper function which bounds attributes of tfa.image to self.
        """
        _ = [
            setattr(self, attrib[0], attrib[1])
            for attrib in inspect.getmembers(tfa.image)
            if inspect.isfunction(attrib[1])
        ]


class Augment(Augmentation):
    """Augment.
    Augmentation Interface which performs the augmentation in pipeline
    in sequential manner.
    """

    def __init__(self, config: Dict, datatype: Text = "bbox"):
        """__init__.

        Args:
            config: config file.
            datatype: dataset type i.e bbox,keypoints,caetgorical,etc.
        """
        self.config = config
        self.transformations = self.config.transformations
        super().__init__(config, self.transformations, type=datatype)
        self.dataset_type = datatype

    def __call__(
        self,
        image: tf.Tensor,
        label: tf.Tensor,
        image_id=None,
        classes=None,
        return_image_label=True,
    ) -> (tf.Tensor, tf.Tensor):
        """__call__.
                Callable which is invoked in tfdata pipeline and performs the
                actual transformation on images and labels.

        Args:
            image: Image Tensor tf.Tensor.
            label: Label tensor tf.Tensor.

        Returns:
            Returns the transform image and labels.
        """
        for transform in self._pipeline:
            transform_func, pass_label = transform
            if pass_label:
                image, label = transform_func(image, label)
            else:
                image = transform_func(image)
        if return_image_label:
            return image, label

        return image_id, image, label, classes
