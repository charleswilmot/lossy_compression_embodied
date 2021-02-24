import tensorflow as tf
from tensorflow import keras
from dataset import get_batched_dataset, get_mean_std_frame
import sys


input_key = 'frame'
output_key = 'frame'
n_epochs = int(sys.argv[3])
z_score_frames = False
path = sys.argv[1]
batch_size = int(sys.argv[2])
dataset = get_batched_dataset(path, batch_size, 1, z_score_frames=z_score_frames)
mean_frame, std_frame = get_mean_std_frame(path)
element = next(iter(dataset))
input_shape = element[input_key].shape[1:]
output_shape = element[output_key].shape[1:]
dataset = dataset.map(lambda x: (x[input_key], x[output_key]))


NFILTERS_0 = 64
FILTER_SIZE_0 = 4
STRIDE_0 = FILTER_SIZE_0 // 2
NFILTERS_1 = 64
FILTER_SIZE_1 = 4
STRIDE_1 = FILTER_SIZE_1 // 2
USE_BIAS = True

model = keras.models.Sequential([
    keras.Input(shape=input_shape),
    keras.layers.Conv2D(NFILTERS_0, FILTER_SIZE_0, STRIDE_0, activation=tf.nn.leaky_relu, padding='same'),
    keras.layers.Conv2D(NFILTERS_1, FILTER_SIZE_1, STRIDE_1, activation=tf.nn.leaky_relu, padding='same'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation=tf.nn.leaky_relu, use_bias=USE_BIAS),
    keras.layers.Dense(8192, activation=tf.nn.leaky_relu, use_bias=USE_BIAS),
    keras.layers.Reshape((8, 16, NFILTERS_1)),
    keras.layers.Conv2DTranspose(NFILTERS_0, FILTER_SIZE_1, STRIDE_1, activation=tf.nn.leaky_relu, padding='same', use_bias=USE_BIAS),
    keras.layers.Conv2DTranspose(3, FILTER_SIZE_0, STRIDE_0, padding='same', use_bias=USE_BIAS),
])



print('at  1', model.layers[1].get_output_shape_at(0))
print('at  2', model.layers[2].get_output_shape_at(0))
print('at -1', model.layers[-1].get_output_shape_at(0))


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['mse']
)
model.fit(dataset, epochs=n_epochs, verbose=1)


import matplotlib.pyplot as plt
import numpy as np
if z_score_frames:
    reconstructions = np.clip(model(element['frame']) * (std_frame + 0.2) + mean_frame, 0, 255).astype(np.uint8)
else:
    reconstructions = np.clip(model(element['frame']) * 127.5 + 127.5, 0, 255).astype(np.uint8)
for reconstruction in reconstructions:
    plt.imshow(reconstruction)
    plt.show()
