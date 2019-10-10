from keras.models import load_model
import matplotlib.pyplot as plt
model = load_model('model_cnn.h5')
model.summary()

from keras.models import Model
weights, biases = model.layers[0].get_weights()
print(weights)
print(len(weights[0]))
print(len(weights))

import cv2

im_gray = cv2.imread('2.jpg', 0)
im_gray = cv2.resize(im_gray, (28, 28)) 
im_gray = 255-cv2.threshold(im_gray, 100, 255, cv2.THRESH_BINARY)[1]
plt.imshow(im_gray)
plt.show()
import numpy as np
x=np.array(im_gray)
#x=x.reshape(1,784)
x = x.astype('float32')
x/=255
y=model.predict(x.reshape(1,28,28,1))
print(y)



layer_outputs = [layer.output for layer in model.layers]
activation_model = Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(x.reshape(1,28,28,1))
 
def display_activation(activations, col_size, row_size, act_index): 
    activation = activations[act_index]
    activation_index=0
    fig, ax = plt.subplots(row_size, col_size, figsize=(row_size*2.5,col_size*1.5))
    for row in range(0,row_size):
        for col in range(0,col_size):
            ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
            activation_index += 1
    plt.show()
    
plt.imshow(x);
plt.show()
display_activation(activations, 5, 4, 1)
