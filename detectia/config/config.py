""" Configuration utility class. """
import yaml
from typing import Text, List, Dict
import copy
import tensorflow as tf


class Config:
    """ Configuration class. """

    def __init__(self):
        # -------dataloader--------
        self.drop_remainder = True
        self.batch_size = 1
        self.transformations = {'flip_left_right': None}
        self.shard = 32
        self.max_instances_per_image = 100

        # --------anchors----------
        self.anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [
            62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]

        # individual grid dimension i.e default [13, 26, 52]
        self.grids = [52, 26, 13]

        # num scale levels i.e default 3.
        # NOTE. len(grids == num_scales
        self.num_scales = 3

        # --------input----------
        self.input_image_shape = (416, 416)

        # -------classes---------
        self.num_classes = 2

    def __setattr__(self, k, v):
        self.__dict__[k] = v

    def __getattr__(self, k):
        return self.__dict__[k]

    def __getitem__(self, k):
        return self.__dict__[k]

    def __repr__(self):
        return repr(self.as_dict())

    def __str__(self):
        print('Configurations:\n')
        try:
            return yaml.dump(self.as_dict(), indent=4)
        except TypeError:
            return str(self.as_dict())

    def update(self, config_dict):
        for key, value in config_dict.items():
            try:
                setattr(self, key, value)
            except KeyError:
                raise f'{key} not present in config class'

    def parse_from_yaml(self, yaml_file_path: Text):
        """Parses a yaml file and returns a dictionary."""
        with tf.io.gfile.GFile(yaml_file_path, 'r') as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
            return config_dict

    def override(self, config):
        """ update config with new configurations. """
        if isinstance(config, str):
            if not config:
                return
            elif config.endswith('.yaml'):
                config_dict = self.parse_from_yaml(config)
            else:
                raise ValueError(
                    'Invalid string {}, must end with .yaml.'.format(
                        config))
        else:
            raise ValueError('Unknown value type: {}'.format(config))

        self.update(config_dict)

    def as_dict(self):
        """Returns a dict representation."""
        config_dict = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                config_dict[k] = v.as_dict()
            else:
                config_dict[k] = copy.deepcopy(v)
        return config_dict
