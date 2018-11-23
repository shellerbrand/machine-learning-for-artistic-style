# Copyright 2018 Stephan Hellerbrand
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generates stylized images with different strengths of a stylization. 
This is a modified version of arbitrary_image_stylization_with_weights.py 
and contains the wrapper function that can be called from within an iPython 
notebook.

For each pair of the content and style images this script computes stylized
images with different strengths of stylization (interpolates between the
identity transform parameters and the style parameters for the style image) and
saves them to the given output_dir.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import os

import numpy as np
import tensorflow as tf

import lib.arbitrary_image_stylization_build_model as build_model
import lib.image_utils as image_utils

slim = tf.contrib.slim

# flags = tf.flags
# flags.DEFINE_string('checkpoint', None, 'Path to the model checkpoint.')
# flags.DEFINE_string('style_images_paths', None, 'Paths to the style images'
#                     'for evaluation.')
# flags.DEFINE_string('content_images_paths', None, 'Paths to the content images'
#                     'for evaluation.')
# flags.DEFINE_string('output_dir', None, 'Output directory.')
# flags.DEFINE_integer('image_size', 256, 'Image size.')
# flags.DEFINE_boolean('content_square_crop', False, 'Wheather to center crop'
#                      'the content image to be a square or not.')
# flags.DEFINE_integer('style_image_size', 256, 'Style image size.')
# flags.DEFINE_boolean('style_square_crop', False, 'Wheather to center crop'
#                      'the style image to be a square or not.')
# flags.DEFINE_integer('maximum_styles_to_evaluate', 1024, 'Maximum number of'
#                      'styles to evaluate.')
# flags.DEFINE_string('interpolation_weights', '[1.0]', 'List of weights'
#                     'for interpolation between the parameters of the identity'
#                     'transform and the style parameters of the style image. The'
#                     'larger the weight is the strength of stylization is more.'
#                     'Weight of 1.0 means the normal style transfer and weight'
#                     'of 0.0 means identity transform.')
# FLAGS = flags.FLAGS

class StyleTransferConfig:
    def __init__(self, checkpoint=None,style_images_paths=None,content_images_paths=None,output_dir=None, image_size=256, content_square_crop=False, style_image_size=256, style_square_crop=False, maximum_styles_to_evaluate=10, interpolation_weights="[1.0]"):
        self.checkpoint = checkpoint
        self.style_images_paths = style_images_paths
        self.content_images_paths = content_images_paths
        self.output_dir = output_dir
        self.image_size = image_size
        self.content_square_crop = content_square_crop
        self.style_image_size = style_image_size
        self.style_square_crop = style_square_crop
        self.maximum_styles_to_evaluate = maximum_styles_to_evaluate
        self.interpolation_weights = interpolation_weights
    
    def __repr__(self):
        xstrlambda = lambda s: '' if s is None else str(s)
        myrep = ""
        myrep = myrep + "checkpoint: {}\n".format(xstrlambda(self.checkpoint))
        myrep = myrep + "style_images_paths: {}\n".format(xstrlambda(self.style_images_paths))
        myrep = myrep + "content_images_paths: {}\n".format(xstrlambda(self.content_images_paths))
        myrep = myrep + "output_dir: {}\n".format(xstrlambda(self.output_dir))
        myrep = myrep + "image_size: {}\n".format(xstrlambda(self.image_size))
        myrep = myrep + "content_square_crop: {}\n".format(xstrlambda(self.content_square_crop))
        myrep = myrep + "style_image_size: {}\n".format(xstrlambda(self.style_image_size))
        myrep = myrep + "style_square_crop: {}\n".format(xstrlambda(self.style_square_crop))
        myrep = myrep + "maximum_styles_to_evaluate: {}\n".format(xstrlambda(self.maximum_styles_to_evaluate))
        myrep = myrep + "interpolation_weights: {}\n".format(xstrlambda(self.interpolation_weights))
        return myrep


##
def arbitrary_stylization_with_weights(stconfig):
  tf.logging.set_verbosity(tf.logging.INFO)
  if not tf.gfile.Exists(stconfig.output_dir):
    tf.gfile.MkDir(stconfig.output_dir)

  with tf.Graph().as_default(), tf.Session() as sess:
    # Defines place holder for the style image.
    style_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
    if stconfig.style_square_crop:
      style_img_preprocessed = image_utils.center_crop_resize_image(
          style_img_ph, stconfig.style_image_size)
    else:
      style_img_preprocessed = image_utils.resize_image(style_img_ph,
                                                        stconfig.style_image_size)

    # Defines place holder for the content image.
    content_img_ph = tf.placeholder(tf.float32, shape=[None, None, 3])
    if stconfig.content_square_crop:
      content_img_preprocessed = image_utils.center_crop_resize_image(
          content_img_ph, stconfig.image_size)
    else:
      content_img_preprocessed = image_utils.resize_image(
          content_img_ph, stconfig.image_size)

    # Defines the model.
    stylized_images, _, _, bottleneck_feat = build_model.build_model(
        content_img_preprocessed,
        style_img_preprocessed,
        trainable=False,
        is_training=False,
        inception_end_point='Mixed_6e',
        style_prediction_bottleneck=100,
        adds_losses=False)

    if tf.gfile.IsDirectory(stconfig.checkpoint):
      checkpoint = tf.train.latest_checkpoint(stconfig.checkpoint)
    else:
      checkpoint = stconfig.checkpoint
      tf.logging.info('loading latest checkpoint file: {}'.format(checkpoint))

    init_fn = slim.assign_from_checkpoint_fn(checkpoint,
                                             slim.get_variables_to_restore())
    sess.run([tf.local_variables_initializer()])
    init_fn(sess)

    # Gets the list of the input style images.
    style_img_list = tf.gfile.Glob(stconfig.style_images_paths)

    print("Getting the list of input style images")
    print(style_img_list)
    

    if len(style_img_list) > stconfig.maximum_styles_to_evaluate:
      np.random.seed(1234)
      style_img_list = np.random.permutation(style_img_list)
      style_img_list = style_img_list[:stconfig.maximum_styles_to_evaluate]

    # Gets list of input content images.
    content_img_list = tf.gfile.Glob(stconfig.content_images_paths)

    print("Getting the list of content images")
    print(content_img_list)
    for content_i, content_img_path in enumerate(content_img_list):
      content_img_np = image_utils.load_np_image_uint8(content_img_path)[:, :, :
                                                                         3]
      content_img_name = os.path.basename(content_img_path)[:-4]

      # Saves preprocessed content image.
      inp_img_croped_resized_np = sess.run(
          content_img_preprocessed, feed_dict={
              content_img_ph: content_img_np
          })
      image_utils.save_np_image(inp_img_croped_resized_np,
                                os.path.join(stconfig.output_dir,
                                             'content_%s.jpg' % (content_img_name)))

      # Computes bottleneck features of the style prediction network for the
      # identity transform.
      identity_params = sess.run(
          bottleneck_feat, feed_dict={style_img_ph: content_img_np})

      for style_i, style_img_path in enumerate(style_img_list):
        if style_i > stconfig.maximum_styles_to_evaluate:
          break
        style_img_name = os.path.basename(style_img_path)[:-4]
        style_image_np = image_utils.load_np_image_uint8(style_img_path)[:, :, :
                                                                         3]

        if style_i % 10 == 0:
          tf.logging.info('Stylizing (%d) %s with (%d) %s' %
                          (content_i, content_img_name, style_i,
                           style_img_name))

        # Saves preprocessed style image.
        style_img_croped_resized_np = sess.run(
            style_img_preprocessed, feed_dict={
                style_img_ph: style_image_np
            })
        image_utils.save_np_image(style_img_croped_resized_np,
                                  os.path.join(stconfig.output_dir,
                                               'style_%s.jpg' % (style_img_name)))

        # Computes bottleneck features of the style prediction network for the
        # given style image.
        style_params = sess.run(
            bottleneck_feat, feed_dict={style_img_ph: style_image_np})

        interpolation_weights = ast.literal_eval(stconfig.interpolation_weights)
        # Interpolates between the parameters of the identity transform and
        # style parameters of the given style image.
        for interp_i, wi in enumerate(interpolation_weights):
          stylized_image_res = sess.run(
              stylized_images,
              feed_dict={
                  bottleneck_feat:
                      identity_params * (1 - wi) + style_params * wi,
                  content_img_ph:
                      content_img_np
              })

          # Saves stylized image.
          image_utils.save_np_image(
              stylized_image_res,
              os.path.join(stconfig.output_dir, 'zzResult_%s_stylized_%s_%d.jpg' %
                           (content_img_name, style_img_name, interp_i)))    
