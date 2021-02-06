""" Box utility classes. """

import functools

import numpy as np
import tensorflow as tf
from .utils import compute_iou_boxes


class BoxEncoder:
    """ a BBox encoder class. """

    def __init__(self, config):
        self.config = config
        self.anchors = config.anchors
        self.input_image_shape = config.input_image_shape
        self.grids = config.grids
        self.num_classes = config.num_classes

        self.num_anchors = len(self.anchors) // len(self.grids)
        self.num_scales = config.level

    @staticmethod
    def _assign_grid(box, grid):
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
    def compute_targets(self, boxes, class_ids):

        # assert num of anchors are compatible with num_scales.
        anchor_ratio = self.num_anchors % self.num_scales
        assert anchor_ratio == 0, "Each feature scale should have same num anchors."

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
            [normalized_boxes_xy, normalized_boxes_wh, tf.ones(shape=(tf.shape(class_ids)[0], 1)), one_hot_classes], axis=-1
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
        targets = []
        for i in range(self.num_scales):

            # grid shape i.e (13, 26, 52, etc)
            grid = self.grids[i]

            # init zero array target for the level.
            target_lvl = tf.zeros(
                shape=(grid, grid, self.num_anchors, 5 + self.num_classes), dtype=tf.float32
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

            targets.append(target_lvl)
        return targets
