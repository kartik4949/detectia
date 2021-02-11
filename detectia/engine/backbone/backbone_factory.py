"""Backbone network factory."""
import os

from absl import logging
import tensorflow as tf

from .efficientnet import Model, get_model_params


def get_model(model_name, override_params=None, model_dir=None):
    """A helper function to create and return model.
    Args:
      model_name: string, the predefined model name.
      override_params: A dictionary of params for overriding. Fields must exist in
        efficientnet_model.GlobalParams.
      model_dir: string, optional model dir for saving configs.
    Returns:
      created model
    Raises:
      When model_name specified an undefined model, raises NotImplementedError.
      When override_params has invalid fields, raises ValueError.
    """

    # For backward compatibility.
    if override_params and override_params.get('drop_connect_rate', None):
        override_params['survival_prob'] = 1 - override_params['drop_connect_rate']

    if not override_params:
        override_params = {}

    if not model_name.startswith('efficientnet-'):
        raise ValueError('Unknown model name {}'.format(model_name))

    blocks_args, global_params = get_model_params(model_name,
                                                  override_params)

    if model_dir:
        param_file = os.path.join(model_dir, 'model_params.txt')
        if not tf.io.gfile.exists(param_file):
            if not tf.io.gfile.exists(model_dir):
                tf.io.gfile.mkdir(model_dir)
            with tf.io.gfile.GFile(param_file, 'w') as f:
                logging.info('writing to %s', param_file)
                f.write('model_name= %s\n\n' % model_name)
                f.write('global_params= %s\n\n' % str(global_params))
                f.write('blocks_args= %s\n\n' % str(blocks_args))

    return Model(blocks_args, global_params, model_name)
