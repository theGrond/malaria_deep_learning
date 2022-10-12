import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import cv2
import tensorflow as tf

from tensorflow import keras

img_height = 80
img_width = 80
IMG_DIMS = (80, 80)

# Set class names
class_names = ['healthy', 'malaria']

# load model
model = tf.keras.models.load_model('basic_cnn.h5')
# model.summary()

# test images path
test_url = 'test/'

# test image name
sunflower_path = test_url+'C241NThinF_IMG_20151207_124608_cell_61.png' 

# load image to keras
img = tf.keras.utils.load_img(
    sunflower_path, target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img) / 255
img_array = tf.expand_dims(img_array, axis=0) # Create a batch

# prediction on image
predictions = model.predict(img_array)
score = float(predictions[0][0])

class_index = int(round(score))
score_percent = score if class_index == 1 else (1-score)

# show result
print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[class_index], 100 * score_percent)
)