# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Evaluation for CIFAR-10.

Accuracy:
task.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by task.py.

Speed:
On a single Tesla K40, task.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from trainer import model
from trainer import cifar10_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', '/home/uc3m4/Documentos/Trained/CIFAR-10',
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('total_steps', 1500,
                            """Number of images that are predicted.""")
tf.app.flags.DEFINE_integer('offset', 0,
                            """Where to start reading in the file.""")
tf.app.flags.DEFINE_string('data_file_bin_path', '/tmp/cifar10_data/cifar-10-batches-bin/test_batch.bin',
                           """Path to the binary file where the images are.""")


def predict(image_number=1):
    with tf.Graph().as_default() as graph:
        image, label = cifar10_input.input_for_prediction_bin(FLAGS.data_file_bin_path, image_number)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = model.inference(image)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))

            result = False
            if sess.run(tf.nn.top_k(logits).indices[0]) == sess.run(label):
                result = True

    return result


def main(argv=None):  # pylint: disable=unused-argument
    model.maybe_download_and_extract()

    total = FLAGS.total_steps
    correct = 0
    offset = FLAGS.offset

    # Run the prediction total_steps times
    for image_number in range(0, total):
        if predict(image_number+offset):
            correct += 1
        if not image_number % 10:
            print("Step", image_number, "of", total)

    print("% correct:", correct/total)


if __name__ == '__main__':
    tf.app.run()
