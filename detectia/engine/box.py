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
""" Box utility classes. """

import functools
from typing import Dict, Text, Tuple

import tensorflow as tf

from ..config import Config
from .utils import compute_iou_boxes


class BoxEncoder:
    """ a BBox encoder class. """

    def __init__(self, config: Config):
        self.config = config
        self.anchors = config.anchors
        self.input_image_shape = config.input_image_shape
        self.grids = config.grids
        self.num_classes = config.num_classes

        self.num_anchors = len(self.anchors) // len(self.grids)
        self.num_scales = config.num_scales

    @staticmethod
    def _assign_grid(box, grid):
        """ helper utility class. """
        return tf.math.floor(box[:, 0] * grid), tf.math.floor(box[:, 1] * grid)

    @staticmethod
    def _best_anchors(boxes, anchors):
        """_best_anchors.

        Args:
            boxes   : pseudo bounding boxes (N, 4).
            anchors : pseudo anchor boxes   (A, 4).

        Returns:
            Intersection over union matrix  (N, A).
        """

        def _compute_iou_along_boxes(box):
            nonlocal anchors
            _compute_iou = functools.partial(compute_iou_boxes, b2=box)
            return tf.map_fn(_compute_iou, anchors)

        return tf.map_fn(_compute_iou_along_boxes, boxes)

    @tf.function
    def compute_targets(self, boxes: tf.Tensor, class_ids: tf.Tensor) -> Dict:
        """compute_targets.
        Computes targets for each scale level.

        Args:
            boxes     : boxes relative to input image. (N, x1, y1, x2, y2).
            class_ids : class_ids (N,)
        Returns:
            list of individual scale targets [(grid, grid, A, O, C)*num_scales].
        """

        # assert num of anchors are compatible with num_scales.
        anchor_ratio = self.num_anchors % self.num_scales
        assert anchor_ratio == 0, "Each feature scale should have same num anchors."
        # indices should be integer.
        class_ids = tf.cast(class_ids, tf.int32)

        # calculate centroid (cx,cy) and width,height of bboxes w.r.t image.
        bb_xy = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
        bb_wh = boxes[:, 2:4] - boxes[:, 0:2]

        # normalize i.e 0-1 range.
        normalized_boxes_xy = bb_xy / self.input_image_shape
        normalized_boxes_wh = bb_wh / self.input_image_shape

        # one hot encoding classes.
        one_hot_classes = tf.one_hot(class_ids, self.num_classes)

        # final true normalize boxes with objectness and class i.e (x, y, w, h, o, c)
        normalized_boxes = tf.concat(
            [
                normalized_boxes_xy,
                normalized_boxes_wh,
                tf.ones(shape=(tf.shape(class_ids)[0], 1)),
                one_hot_classes,
            ],
            axis=-1,
        )

        # convert (wh) to (0,0,w,h) points
        pseudo_bboxes = tf.concat([tf.zeros(shape=tf.shape(bb_wh)), bb_wh], axis=-1)
        pseudo_anchor_boxes = tf.concat(
            [tf.zeros(shape=tf.shape(self.anchors)), self.anchors], axis=-1
        )

        # get the iou matrix for anchors.
        _best_anchors = self._best_anchors(pseudo_bboxes, pseudo_anchor_boxes)

        # top iou anchor id.
        _best_anchors_ids = tf.argmax(_best_anchors, axis=-1)
        _best_anchors_ids = tf.cast(_best_anchors_ids, tf.int32)

        # targets i.e level 1, 2, 3, etc.
        targets = {}
        for i in range(self.num_scales):

            # grid shape i.e (13, 26, 52, etc)
            grid = self.grids[i]

            # init zero array target for the level.
            target_lvl = tf.zeros(
                shape=(grid, grid, self.num_anchors, 5 + self.num_classes),
                dtype=tf.float32,
            )

            # calculate anchors for the current scale_level i.e (1, 2, 3)
            lower_bound = tf.cast(
                tf.greater(i * self.num_anchors - 1, _best_anchors_ids), dtype=tf.int32
            )
            upper_bound = tf.cast(
                tf.greater(_best_anchors_ids, i * self.num_anchors + self.num_anchors),
                dtype=tf.int32,
            )
            anchors_level_ids = tf.math.logical_not(
                tf.cast(lower_bound + upper_bound, tf.bool)
            )

            # get the best match boxes with anchors in the level.
            best_boxes_lvl = tf.boolean_mask(normalized_boxes, anchors_level_ids)
            best_boxes_lvl = tf.cast(best_boxes_lvl, tf.float32)

            # anchors ids at the level.
            best_anchors_ids_lvl = tf.boolean_mask(_best_anchors_ids, anchors_level_ids)

            # get grid box step for the matched boxes .
            gi, gj = self._assign_grid(best_boxes_lvl, grid)

            # update the target level array at matched anchor id with best boxes.
            idx = tf.stack(
                [
                    tf.cast(gi, tf.int32),
                    tf.cast(gj, tf.int32),
                    tf.cast(best_anchors_ids_lvl, tf.int32),
                ]
            )
            idx = tf.transpose(idx)
            target_lvl = tf.tensor_scatter_nd_update(
                target_lvl, [idx % self.num_anchors], [best_boxes_lvl]
            )

            targets.update({f"scale_level_{i+1}": target_lvl})
        return targets


class BoxDecoder:
    """ BoxDecoder utility class. """

    def __init__(self, config: Config):
        self.config = config
        self.input_shape = config.input_image_shape

    @tf.function
    def decode_model_features(self, features: tf.Tensor, anchors: tf.Tensor) -> Tuple:
        """decode_model_features.
        Decodes feature ouputs from model.

        Args:
            features : model feaure ouput (B, Gi, Gj, A, (5 + C)).
            anchors  : anchors for the feature.

        Returns:
            True box_xy, box_wh confidence and class probs.
        """
        grid_shape = tf.shape(features)[1:3]
        anchors = tf.reshape(tf.constant(anchors), [1, 1, 1, len(anchors), 2])

        # create grid tensor with relative cx, cy as values.
        grid_x = tf.tile(
            tf.reshape(tf.range(0, grid_shape[0]), shape=[grid_shape[0], 1, 1, 1]),
            [1, grid_shape[0], 1, 1],
        )
        grid_y = tf.tile(
            tf.reshape(tf.range(0, grid_shape[1]), shape=[1, grid_shape[1], 1, 1]),
            [grid_shape[1], 1, 1, 1],
        )
        grid_cells = tf.cast(tf.concat([grid_x, grid_y], axis=-1), tf.float32)

        # Yolov3 https://arxiv.org/abs/1804.02767
        # bx = sigmoid(tx) + cx
        # bh = e^ph * th
        box_xy = tf.nn.sigmoid(features[..., :2]) + grid_cells
        box_xy = box_xy / tf.cast(grid_shape[..., ::-1], features.dtype)
        box_wh = tf.exp(features[..., 2:4]) * tf.cast(anchors, tf.float32)
        # TODO reverse the input_shape.
        box_wh = box_wh / tf.cast(self.input_shape, features.dtype)

        # confidence and class probs
        confidence = tf.nn.sigmoid(features[..., 4])
        class_probs = tf.nn.sigmoid(features[..., 5:])
        return box_xy, box_wh, confidence, class_probs
