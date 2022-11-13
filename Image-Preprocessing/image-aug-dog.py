#https://neptune.ai/blog/what-image-processing-techniques-are-actually-used-in-the-ml-industry

import matplotlib.pyplot as plt #For plotting our visualizations
from keras.preprocessing.image import ImageDataGenerator #Keras dataset generator class.
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
#import cv2
from PIL import Image

# %matplotlib inline  # Uncomment this line if running this code in Jypiter notebook

image = Image.open('dog.png')
# plt.imshow(image)
# plt.show()

#Rotation
data = img_to_array(image)
samples = expand_dims(data, 0)
data_generated = ImageDataGenerator(rotation_range=90)  #Here we specify angle of rotation as 90
it = data_generated.flow(samples, batch_size=1)

for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    result = batch[0].astype('uint8')
    plt.imshow(result)
#plt.show()


# Translation

data = img_to_array(image)
samples = expand_dims(data, 0)
data_generator = ImageDataGenerator(height_shift_range=0.5)
it = data_generator.flow(samples, batch_size=1)
for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    result = batch[0].astype('uint8')
    plt.imshow(result)
#plt.show()


data = img_to_array(image)
samples = expand_dims(data, 0)
data_generator = ImageDataGenerator(width_shift_range=[-100,100])
it = data_generator.flow(samples, batch_size=1)
for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    result = batch[0].astype('uint8')
    plt.imshow(result)
# plt.show()

# color space
data = img_to_array(image)
samples = expand_dims(data, 0)
datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
it = datagen.flow(samples, batch_size=1)
for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    result = batch[0].astype('uint8')
    plt.imshow(result)
#plt.show()

#Zooming

data = img_to_array(image)
samples = expand_dims(data, 0)
datagen = ImageDataGenerator(zoom_range=[0.2,1.0])
it = datagen.flow(samples, batch_size=1)
for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    result = batch[0].astype('uint8')
    plt.imshow(result)
#plt.show()

#Flipping
data = img_to_array(image)
samples = expand_dims(data, 0)
datagen = ImageDataGenerator(vertical_flip=True)
it = datagen.flow(samples, batch_size=1)
for i in range(9):
    plt.subplot(330 + 1 + i)
    batch = it.next()
    result = batch[0].astype('uint8')
    plt.imshow(result)
plt.show()