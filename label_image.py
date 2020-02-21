# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""label_image for tflite."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np

from PIL import Image

#import tensorflow as tf # TF2
import tflite_runtime.interpreter as tflite

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '-i',
      '--image',
      default='/tmp/test1.jpg',
      help='image to be classified')
  parser.add_argument(
      '-m',
      '--model_file',
      default='/tmp/detect.tflite',
      help='.tflite model to be executed')
  parser.add_argument(
      '-l',
      '--label_file',
      default='/tmp/labels.txt',
      help='name of file containing labels')
  parser.add_argument(
      '--input_mean',
      default=128, type=np.uint8,
      help='input_mean')
  parser.add_argument(
      '--input_std',
      default=128, type=np.uint8,
      help='input standard deviation')
  args = parser.parse_args()

  interpreter = tflite.Interpreter(model_path=args.model_file)
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # print input and output details
  print("==Input Data==")
  print("shape: ", input_details[0]['shape'])

  # check the type of the input tensor
  floating_model = input_details[0]['dtype'] == np.float32

  print("floating_model : ", floating_model)

  quant_model = input_details[0]['dtype'] == np.uint8

  print("quant_model : ", quant_model)

  # NxHxWxC, H:1, W:2
  height = input_details[0]['shape'][1]
  width = input_details[0]['shape'][2]
  img = Image.open(args.image).resize((width, height))

  # add N dim
  input_data = np.expand_dims(img, axis=0)

  if floating_model:
    input_data = (np.float32(input_data) - args.input_mean) / args.input_std

  interpreter.set_tensor(input_details[0]['index'], input_data)

  interpreter.invoke()

  output_data = interpreter.get_tensor(output_details[0]['index'])
  results = np.squeeze(output_data)

  top_k = results.argsort()[-5:][::-1]
  labels = load_labels(args.label_file)
  for i in top_k:
    if floating_model:
      # print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
      print('this is the floating model results section')
    else:
      # print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
      # print('{:08.6f}: {}'.format(float(results[i]/255.0), labels[i]))
      print(results[i])
      print(labels[i])