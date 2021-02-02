import tensorflow as tf


class Anchor:
    def __init__(self, config):
        self.config = config

    def _generate_anchors(
        self,
    ):
        ...

    def compute_targets(self, boxes, class_ids):
        ...
