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
""" engine utilities helper functions. """
import tensorflow as tf


def compute_iou_boxes(b1, b2):
    # determine the coordinates of the intersection rectangle
    x_left = tf.math.maximum(b1[0], b2[0])
    y_top = tf.math.maximum(b1[1], b2[1])
    x_right = tf.math.minimum(b1[2], b2[3])
    y_bottom = tf.math.minimum(b1[3], b2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # the intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both aabbs
    bb1_area = (b1[2] - b1[0]) * (b1[3] - b1[1])
    bb2_area = (b2[2] - b2[0]) * (b2[3] - b2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / (bb1_area + bb2_area - intersection_area)
    return iou
