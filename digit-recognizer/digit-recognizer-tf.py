import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

sns.set_style("whitegrid")

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))


ITERATOR_BATCH_SIZE = 100
column_defaults = [tf.int32]*785
train = tf.data.experimental.make_csv_dataset('train.csv', batch_size=10, label_name='label' ,header=True)
test = tf.data.experimental.make_csv_dataset('test.csv', batch_size=10,header=True)


model = tf.keras.Sequential([
# Adds a densely-connected layer with 64 units to the model:
layers.Dense(784, activation='relu', input_shape=(28,28,1)),
# Add another:
layers.Dense(784, activation='relu'),
# Add a softmax layer with 10 output units:
layers.Dense(10, activation='softmax')])



model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

sample_count = 42000
batch_size = 1
steps_per_epoch = sample_count // batch_size
model.fit(train, steps_per_epoch=steps_per_epoch,epochs=1)