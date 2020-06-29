from keras.layers import Dense
from keras.layers import Reshape, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Sequential
from keras.datasets import mnist
import numpy as np

from dehydrated_vae import build_vae

#preprocess mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

#create encoder and decoder
#NOTE: the encoder does not contain the latent mean/stddev layers
latent_size = 2
acti = "tanh"
encoder = Sequential([
  Dense(256, input_shape=[28 * 28], activation=acti),
  Dense(128, activation=acti)
])

decoder = Sequential([
  Dense(256, input_shape=[latent_size]),
  Dense(128, activation=acti),
  Dense(28 * 28, activation="sigmoid")
])

#create the VAE
#the encoder will be wrapped in a new model containing the latent mean layer
vae, encoder, decoder, loss = \
  build_vae(encoder, decoder, latent_size, kl_scale=1/np.prod(x_train.shape[1:]))

vae.compile(optimizer="adam", loss=loss)

vae.summary()

vae.fit(x_train, x_train, epochs=10)

#####

import matplotlib.pyplot as plt

n = 20
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
xyrange = 1
grid_x = np.linspace(-xyrange, xyrange, n)
grid_y = np.linspace(-xyrange, xyrange, n)

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()

#####

y_map = encoder.predict(x_test)
plt.figure(figsize=(10, 10))
plt.scatter(y_map[:, 0], y_map[:, 1], c=y_test, cmap="jet", edgecolors="k")
plt.colorbar()
plt.show()
