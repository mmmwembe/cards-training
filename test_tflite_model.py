
import numpy as np
import tensorflow as tf
from tensorflow.lite.python import interpreter as interpreter_wrapper
import argparse

from PIL import Image

# Variable for Model and Image File
TEST_IMAGE_FILE ="images/real_test/test1.jpg"
TFLITE_QUANT_MODEL="detector/detect.tflite"


# Load TFLite model and allocate tensors.
interpreter = interpreter_wrapper.Interpreter(model_path=TFLITE_QUANT_MODEL)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# check the type of the input tensor
floating_model = False
quant_model = False

if input_details[0]['dtype'] == np.float32:
    floating_model = True
if input_details[0]['dtype'] == np.uint8:
    quant_model = True
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

img = Image.open( TEST_IMAGE_FILE)
img.load()
# img = Image.open(TEST_IMAGE_FILE)
img = img.resize((width, height))

input_mean = 128
input_std = 128
input_data = np.expand_dims(img, axis=0)

print('floating model: {} '.format(floating_model))
print('quant_model: {}' .format(quant_model))


if floating_model:
  input_data = (np.float32(input_data) - input_mean) / input_std
if quant_model:
  input_data = (np.uint8(input_data) - input_mean) / input_std
  #input_data = (np.int32(input_data) - input_mean) / input_std  # changed from np.unit8 to np.int32 to resolve the following error: ValueError: Cannot set tensor: Got tensor of type 0 but expected type 3 for input 175  resolve the type 

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(input_data, dtype=np.uint8)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

print('INPUTS: ')
print(input_details)
print('OUTPUTS: ')
print(output_details)