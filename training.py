from __future__ import print_function
import keras
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, Activation
from keras.models import Sequential
from keras.constraints import maxnorm
from keras import regularizers
import numpy as np


batch_size = 64
num_classes = 10
epochs = 30

# input image dimensions
img_x, img_y = 28, 28

# load the data
train_data = np.load('train_data.npy')
train_label = np.load('train_label.npy')

# format the data as arrays and split into trianing and testing sets
x_train = train_data[:40000, :]
y_train = train_label[:40000]
x_test = train_data[40000:, :]
y_test = train_label[40000:]

# convert the data to the right type
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# predict /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices - this is for use in the
# categorical_crossentropy loss below
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

decay = 1e-4
input_shape = (img_x, img_y, 3)
model = Sequential()

model.add(Conv2D(32, (2,2), padding='same', kernel_regularizer=regularizers.l2(decay), input_shape=input_shape))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (2,2), padding='same', kernel_regularizer=regularizers.l2(decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (2,2), padding='same', kernel_regularizer=regularizers.l2(decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (2,2), padding='same', kernel_regularizer=regularizers.l2(decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))

model.add(Conv2D(256, (2,2), padding='same', kernel_regularizer=regularizers.l2(decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (2,2), padding='same', kernel_regularizer=regularizers.l2(decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(512, (2,2), padding='same', kernel_regularizer=regularizers.l2(decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(512, (2,2), padding='same', kernel_regularizer=regularizers.l2(decay)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.6))

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))

history = AccuracyHistory()

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[history])
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model.save('model.h5')
