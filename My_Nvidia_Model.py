import json
import numpy as np
import pandas as pd
import cv2
import os
import scipy.misc

import tensorflow as tf
tf.python.control_flow_ops = tf

from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, ELU, Flatten, Lambda
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from PIL import Image
from sklearn.model_selection import train_test_split

# Location of the training data
DATA_FILE = 'driving_log.csv'

print('\n[INFO] Loading Data:')
# Load the training data into a pandas dataframe.
columns = ['Center Image', 'Left Image', 'Right Image', 'Steering Angle', 'Throttle', 'Break', 'Speed']
data = pd.read_csv(DATA_FILE, names=columns, header=1)

num_bins = 23
avg_samples_per_bin = len(data['Steering Angle'])/num_bins
hist, bins = np.histogram(data['Steering Angle'], num_bins)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) * 0.5

print('\n[INFO] Dropping Random Straight Steering Angles:')
keep_probs = []
target = avg_samples_per_bin * .3
for i in range(num_bins):
    if hist[i] < target:
        keep_probs.append(1.)
    else:
        keep_probs.append(1./(hist[i]/target))

remove_list = []
for i in range(len(data['Steering Angle'])):
    for j in range(num_bins):
        if data['Steering Angle'][i] > bins[j] and data['Steering Angle'][i] <= bins[j+1]:
            # delete from X and y with probability 1 - keep_probs[j]
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)
data.drop(data.index[remove_list], inplace=True)

images = data[['Center Image', 'Left Image', 'Right Image']]
angles = data['Steering Angle']

print('\n[INFO] Creating Training and Testing Data:')
images_train, images_validation, angles_train, angles_validation = train_test_split(
    images, angles, test_size=0.15, random_state=42)

print('\n[INFO] Preprocessing Images and Augmenting Data:')
def load_image_and_preprocess(path, flip_image=False, tint_image=False):
    # Open image from disk and flip it if generating data.
    image = Image.open(path.strip())

    if flip_image:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)

    if tint_image:
        image = cv2.imread(path.strip())
        image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
        image = np.array(image, dtype = np.float64)
        random_bright = .5+np.random.uniform()
        image[:,:,2] = image[:,:,2]*random_bright
        image[:,:,2][image[:,:,2]>255]  = 255
        image = np.array(image, dtype = np.uint8)
        image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)

    # Convert the image into mulitdimensional matrix of float values (normally int which messes up our division).
    image = np.array(image, np.float32)

    # Crop Image
    image = image[35:135, :]
    image = scipy.misc.imresize(image, (66,200))

    # Return the modified image.
    return image

def jitter_image(path, steering):
    image = cv2.imread(path.strip())
    rows, cols, _ = image.shape
    transRange = 100
    numPixels = 10
    valPixels = 0.4
    transX = transRange * np.random.uniform() - transRange/2
    steering = steering + transX/transRange * 2 * valPixels
    transY = numPixels * np.random.uniform() - numPixels/2
    transMat = np.float32([[1, 0, transX], [0, 1, transY]])
    image = cv2.warpAffine(image, transMat, (cols, rows))

    # Convert the image into mulitdimensional matrix of float values (normally int which messes up our division).
    image = np.array(image, np.float32)

    # Crop Image
    image = image[35:135, :]
    image = scipy.misc.imresize(image, (66,200))

    return image, steering

print('\n[INFO] Batch Generator:')
def batch_generator(images, steering_angles, batch_size=64, augment_data=True):
    # Create an array of sample indices.
    batch_images = []
    batch_steering_angles = []
    sample_count = 0
    SIDE_STEERING_CONSTANT = 0.25
    indices = np.arange(len(images))

    while True:
        # Shuffle indices to minimize overfitting. Common procedure.
        np.random.shuffle(indices)
        for i in indices:

            # Load the center image and steering angle.
            center_image = load_image_and_preprocess(images.iloc[i]['Center Image'])
            center_angle = float(steering_angles.iloc[i])
            batch_images.append(center_image)
            batch_steering_angles.append(center_angle)

            # Add augmentation if needed. We do this because our model only runs on
            # our center camera feed and we dont want to modify anything other than
            # the cropping and normalizing for our validation dataset since this should
            # work on raw data.
            if augment_data:
                # Flip the image and invert angle respectively.
                flipped_image = load_image_and_preprocess(images.iloc[i]['Center Image'], True, False)
                flipped_angle = -1.0 * center_angle
                batch_images.append(flipped_image)
                batch_steering_angles.append(flipped_angle)

                # Tint the center image to random brightness.
                tint_image = load_image_and_preprocess(images.iloc[i]['Center Image'], False, True)
                tint_angle = center_angle
                batch_images.append(tint_image)
                batch_steering_angles.append(tint_angle)

                # Jitter the center image to make it seem like different position on road.
                jittered_image, jitter_angle = jitter_image(images.iloc[i]['Center Image'], center_angle)
                batch_images.append(jittered_image)
                batch_steering_angles.append(jitter_angle)

                # Load the left image and add steering constant to compensate for shift.
                left_image = load_image_and_preprocess(images.iloc[i]['Left Image'])
                # Steering angle must stay within the range of -1 and 1
                left_angle = min(1.0, center_angle + SIDE_STEERING_CONSTANT)
                batch_images.append(left_image)
                batch_steering_angles.append(left_angle)

                # Load the left image and subtract steering constant to compensate for shift.
                right_image = load_image_and_preprocess(images.iloc[i]['Right Image'])
                # Steering angle must stay within the range of -1 and 1
                right_angle = max(-1.0, center_angle - SIDE_STEERING_CONSTANT)
                batch_images.append(right_image)
                batch_steering_angles.append(right_angle)

            # Increment the number of samples.
            sample_count += 1

            # If we have processed batch_size number samples or this is the last batch
            # of the epoch, then we submit the batch. Since we augment the data there is a chance
            # we have more than the number of batch_size elements in each batch.
            if (sample_count % batch_size) == 0 or (sample_count % len(images)) == 0:
                yield np.array(batch_images), np.array(batch_steering_angles)
                # Reset
                batch_images = []
                batch_steering_angles = []


activation_relu = 'relu'
learning_rate = 1e-4

def create_model():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(66, 200, 3)))
    model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    model.add(Activation(activation_relu))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1164))
    model.add(Activation(activation_relu))
    model.add(Dense(100))
    model.add(Activation(activation_relu))
    model.add(Dense(50))
    model.add(Activation(activation_relu))
    model.add(Dense(10))
    model.add(Activation(activation_relu))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate), loss="mse", )
    return model

nb_epoch = 10

samples_per_epoch = 6 * len(images_train)
generator_train = batch_generator(images_train, angles_train)

nb_val_samples = len(images_validation)
generator_validation = batch_generator(images_validation, angles_validation, augment_data=False)

print('\n[INFO] Creating Model:')
model = create_model()
print('\n[INFO] Training Model:')
model.fit_generator(generator_train,
                              samples_per_epoch=samples_per_epoch,
                              nb_epoch=nb_epoch,
                              validation_data=generator_validation,
                              nb_val_samples=nb_val_samples)

model.save_weights('model.h5', True)
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)