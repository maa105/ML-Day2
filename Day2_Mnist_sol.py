from keras import backend as k
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt

NB_EPOCH = 20
BATCH_SIZE = 128
VERBOSE = 1
NB_CLASSES = 10 # number of outputs = number of digits
OPTIMIZER = Adam() # SGD optimizer, explained later in this chapter
N_HIDDEN = 128
DROPOUT = 0.3
VALIDATION_SPLIT=0.2 # how much TRAIN is reserved for VALIDATION
# data: shuffled and split between train and test sets

(X_train,y_train), (X_test, y_test)= mnist.load_data()

X_train=X_train.astype('float32')
X_test=X_test.astype('float32')

X_train/=255
X_test=X_test/255

X_train=X_train[:,:,:,np.newaxis]
X_test=X_test[:,:,:,np.newaxis]

Y_train=np_utils.to_categorical(y_train,10)
Y_test=np_utils.to_categorical(y_test,10)



model = Sequential()
model.add(Conv2D(20,kernel_size=5,padding="same",input_shape=(28,28,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(50,kernel_size=5,padding="same"))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])

history=model.fit(X_train,Y_train, batch_size=BATCH_SIZE,epochs=NB_EPOCH, verbose=VERBOSE,validation_split=0.2)

score=model.evaluate(X_test,Y_test,verbose=1)
print("Test score:",score[0])
print("Test accuracy:",score[1])

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.legend(['train','validation'], loc='upper left')
plt.show()

model.save('model.h5')
print("Model saved")