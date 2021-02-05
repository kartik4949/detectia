import functools
import tensorflow as tf
import numpy as np
from utils import compute_iou_boxes


class BoxEncoder:
    def __init__(self, config):
        self.config = config
        self.anchors = np.asarray(
            [[10, 20], [30, 40], [50, 60], [200, 100]], np.float32)
        self.input_image_shape = (416, 416)
        self.grids = [13, 26]
        self.num_anchors = 2
        self.num_scale_level = 2

    def _best_anchors(self, boxes, anchors):
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

    def _assign_grid(self, box, grid):
        return tf.math.floor(box[:, 0] * grid), tf.math.floor(box[:, 1] * grid)

    @tf.function
    def compute_targets(self, boxes, class_ids):

        # assert num of anchors are compatible with num_scales.
        # anchor_ratio = self.config.anchors % self.config.num_scales
        anchor_ratio = 0
        assert anchor_ratio == 0, f'Each feature scale should have same num anchors.'

        # calculate centroid (cx,cy) and width,height of bboxes w.r.t image.
        bb_xy = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
        bb_wh = boxes[:, 2:4] - boxes[:, 0:2]

        # normalize i.e 0-1 range.
        normalized_boxes_top = boxes[:, 0:2] / self.input_image_shape
        normalized_boxes_bottom = boxes[:, 2:4] / self.input_image_shape
        normalized_boxes = tf.concat(
            [normalized_boxes_top, normalized_boxes_bottom], axis=-1)

        # convert (wh) to (0,0,w,h) points
        pseudo_bboxes = tf.concat([tf.zeros(shape=tf.shape(bb_wh)), bb_wh], axis=-1)
        pseudo_anchor_boxes = tf.concat(
            [tf.zeros(shape=tf.shape(self.anchors)), self.anchors], axis=-1)

        # get the iou matrix for anchors.
        _best_anchors = self._best_anchors(pseudo_bboxes, pseudo_anchor_boxes)

        # top iou anchor id.
        _best_anchors_ids = tf.argmax(_best_anchors, axis=-1)
        _best_anchors_ids = tf.cast(_best_anchors_ids, tf.int32)

        targets = []
        for i in range(self.num_scale_level):
            grid = self.grids[i]
            lower_bound = tf.cast(tf.greater(
                i * self.num_anchors - 1, _best_anchors_ids), dtype=tf.int32)
            upper_bound = tf.cast(tf.greater(
                _best_anchors_ids, i * self.num_anchors + self.num_anchors), dtype=tf.int32)
            anchors_level_ids = tf.math.logical_not(
                tf.cast(lower_bound + upper_bound, tf.bool))

            best_boxes_lvl = tf.boolean_mask(normalized_boxes, anchors_level_ids)
            best_boxes_lvl = tf.cast(best_boxes_lvl, tf.float32)

            best_anchors_ids_lvl = tf.boolean_mask(_best_anchors_ids, anchors_level_ids)

            gi, gj = self._assign_grid(best_boxes_lvl, grid)

            target_lvl = tf.zeros(
                shape=(grid, grid, self.num_anchors, 4), dtype=tf.float32)

            idx = tf.stack([tf.cast(gi, tf.int32), tf.cast(gj, tf.int32),
                            tf.cast(best_anchors_ids_lvl, tf.int32)])

            idx = tf.transpose(idx)
            target_lvl = tf.tensor_scatter_nd_update(
                target_lvl, [idx % self.num_anchors], [best_boxes_lvl])
            targets.append(target_lvl)
        return targets


if __name__ == "__main__":
    config = None
    boxes = [[200, 100, 300, 200], [10, 20, 30, 40], [28, 38, 40, 50]]
    boxes = np.asarray(boxes, np.float32)
    x = BoxEncoder(config)
    x = x.compute_targets(boxes, None)
    breakpoint()
