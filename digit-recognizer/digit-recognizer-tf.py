import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
from sklearn.model_selection import train_test_split

sns.set_style("whitegrid")


def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


ITERATOR_BATCH_SIZE = 100
column_defaults = [tf.int32] * 785
# train = tf.data.experimental.make_csv_dataset('train.csv', batch_size=32, label_name='label' ,header=True)

# test = tf.data.experimental.make_csv_dataset('test.csv', batch_size=10,header=True)


train = pd.read_csv("train.csv").values
test = pd.read_csv("test.csv").values
X = train[:, 1:].reshape(train.shape[0], 1, 28, 28).astype('float32')
X = X / 255.0
Y = train[:, 0]

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.1,random_state=0)

# (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# x_train = x_train.reshape(60000, 784).astype('float32') / 255
# x_test = x_test.reshape(10000, 784).astype('float32') / 255


inputs = keras.Input(shape=(784,))
dense = layers.Dense(64, activation='relu')
x = dense(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=5,
                    validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=0)
print('History:', history)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])
