#https://prince-canuma.medium.com/image-pre-processing-c1aec0be3edf

"""In this example, we are going to go through the steps of Image preprocessing needed to train,
validate and test any AI-Computer Vision model."""

# step 1 Read Images

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

image_path = ""

def load_images(path):
    image_files = sorted([os.path.join(path, 'train', file) for file in os.listdir(path + '/train') if file.endswith('.png')])
    return image_files


# step 1 Re-size Images

#display one  image
def display_one(a, title1 = "Original"):
    plt.imshow(a)
    plt.title(title1)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# display two images
def display_two(a, b, title1="Original", title2="Edited"):
    plt.subplot(121)
    plt.imshow(a)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(122)
    plt.imshow(b)
    plt.xticks([])
    plt.yticks([])
    plt.show()

# pre-processing
def processing(data):
    # loading image
    # Getting 3 images to work with
    img = [cv2.imread(i, cv2.IMREAD_UNCHANGED) for i in data[:3]]
    print('Original size', img[0].shape)
    # --------------------------------
    # setting dim of the resize
    height = 220
    width = 220
    dim = (width, height)
    res_img = []
    for i in range(len(img)):
        res = cv2.resize(img[i], dim, interpolation=cv2.INTER_LINEAR)
        res_img.append(res)

    # Checcking the size
    print("RESIZED", res_img[1].shape)

    # Visualizing one of the images in the array
    original = res_img[1]
    display_one(original)