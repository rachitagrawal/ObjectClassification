import tensorflow as tf
import matplotlib.pyplot as plt
import time
import math
import os
import keras
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

(images_train, cls_train), (images_test, cls_test) = cifar10.load_data()

print(images_test.shape, cls_test.shape)
class_names = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

print("Size of:")
print("- Training Set:\t\t{}".format(len(images_train)))
print("- Testing Set:\t\t{}".format(len(images_test)))

print(images_train.shape)
num_train, img_size, _, num_channels = images_train.shape
num_classes = len(np.unique(cls_train))

input_shape = (img_size, img_size, num_channels)
print(input_shape)

#fig = plt.figure(figsize=(8,3))
#for i in range(num_classes):
#    ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
#    idx = np.where(cls_train[:]==i)[0]
#    features_idx = images_train[idx,::]
#    img_num = np.random.randint(features_idx.shape[0])
#    im = np.transpose(features_idx[img_num,::], (0, 1, 2))
#    ax.set_title(class_names[i])
#    plt.imshow(im, interpolation='spline16')
#plt.show()

#Normalize each channel values between 0 and 1
images_train = images_train.astype('float32')/255
images_test = images_test.astype('float32')/255
# convert class labels to binary class labels
cls_train = keras.utils.to_categorical(cls_train, num_classes)
cls_test = keras.utils.to_categorical(cls_test, num_classes)

# define model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# compile model
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# print model summary
print(model.summary())

#Create Model checkpoing and CSV Logger
model_checkpoint = keras.callbacks.ModelCheckpoint('cifar10_weights.{epoch:02d}-{val_loss:.2f}.hdf5')
csv_logger = keras.callbacks.CSVLogger('cifar10_training.log')

batch_size = 128
epochs = 1

#Train the model and print out the evaluation metric 'accuracy' for test set
model.fit(images_train, cls_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          callbacks = [model_checkpoint, csv_logger],
          validation_data=(images_test, cls_test))


print("All Done")