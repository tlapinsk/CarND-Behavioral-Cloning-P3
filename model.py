import csv
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from random import shuffle

def resize(data):
    from keras.backend import tf as ktf
    return ktf.image.resize_images(data, (64, 64))

samples = []
with open('/home/carnd/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

del samples[0]
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            adjustment = 0.15
            for batch_sample in batch_samples:
                
                # Read in center, left, and right images
                center_image = '/home/carnd/data/IMG/'+batch_sample[0].split('/')[-1]
                left_image = '/home/carnd/data/IMG/'+batch_sample[1].split('/')[-1]
                right_image = '/home/carnd/data/IMG/'+batch_sample[2].split('/')[-1]
                
                # Grab images and associated angles
                center_image = cv2.imread(center_image)
                center_angle = float(batch_sample[3])
                left_image = cv2.imread(left_image)
                left_angle = float(batch_sample[3]) + adjustment
                right_image = cv2.imread(right_image)
                right_angle = float(batch_sample[3]) - adjustment
                
                # Extend images to lists
                images.extend((center_image, left_image, right_image))
                angles.extend((center_angle, left_angle, right_angle))
            
            # Augment images with flip probability
            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                flip_prob = np.random.random()
                if flip_prob > 0.3 or abs(angle) < 0.05:
                    pass
                else:
                    augmented_images.append(cv2.flip(image, 1))
                    augmented_angles.append(angle*-1.0)

            # Turn into np.array
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Code to create histogram (not used)
# print("Creating histogram...")
# plt.hist(measurements, bins=20)
# plt.show()

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# import Keras and necessary add-ons
import keras
from keras.models import Sequential
from keras import optimizers
from keras.layers import Flatten, Dense, Lambda, Activation, Dropout
from keras.layers import Conv2D, MaxPooling2D, Cropping2D, Reshape
from keras.callbacks import History 

# Nvidia architecture
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))  # Normalize data
model.add(Cropping2D(cropping=((50,20),(0,0))))  # Crop images
model.add(Lambda(resize))  # Resize to 64x64
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Compile and train the model
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
history = model.fit_generator(train_generator, steps_per_epoch=
            len(train_samples)/32, validation_data=validation_generator,
            validation_steps=len(validation_samples)/32, epochs=3)

# Save the model
model.save('model.h5')
print("Model saved")

# Print Keras history keys
print()
print(history.history.keys())

# Summarize loss history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss Plot')
plt.ylabel('MSE Loss')
plt.xlabel('Epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()