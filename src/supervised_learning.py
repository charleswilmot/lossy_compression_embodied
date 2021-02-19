import tensorflow as tf
from tensorflow import keras
from dataset import get_batched_dataset
import sys


input_key = 'frame'
output_key = 'arm0_end_eff'
n_epochs = int(sys.argv[3])

dataset = get_batched_dataset(sys.argv[1], int(sys.argv[2]), 1)
element = next(iter(dataset))
input_shape = element[input_key].shape[1:]
output_shape = element[output_key].shape[1:]
dataset = dataset.map(lambda x: (x[input_key], x[output_key]))

model = keras.models.Sequential([
    keras.Input(shape=input_shape),
    keras.layers.Conv2D(128, 8, 4, activation=tf.nn.relu),
    keras.layers.Conv2D(128, 4, 2, activation=tf.nn.relu),
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(100, activation=tf.nn.relu),
    keras.layers.Dense(output_shape[0]),
])


model.compile(
    optimizer='adam',
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=['mse']
)
model.fit(dataset, epochs=n_epochs, verbose=1)
