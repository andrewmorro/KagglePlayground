import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow import keras

sns.set_style("whitegrid")

def show_batch(dataset):
  for batch, label in dataset.take(1):
    for key, value in batch.items():
      print("{:20s}: {}".format(key,value.numpy()))


ITERATOR_BATCH_SIZE = 100
column_defaults = [tf.int32]*785
train = tf.data.experimental.make_csv_dataset('train.csv', batch_size=32, label_name='label' ,header=True)

test = tf.data.experimental.make_csv_dataset('test.csv', batch_size=10,header=True)

inputs = keras.Input(shape=(784,))
dense = layers.Dense(64, activation='relu')
x = dense(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
             loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2)