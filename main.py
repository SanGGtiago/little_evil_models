from models import *
import numpy as np
import tensorflow as tf

data = np.load('data/similarity.npy', allow_pickle=True)
print(data.shape, data.dtype)

dataset = tf.data.Dataset.from_tensor_slices((data, data))
print(dataset.element_spec)
train_dataset = dataset.take(75)
test_dataset = dataset.skip(75)

model = MLP(train_dataset.element_spec[0].shape)

train_dataset = train_dataset.batch(5)
test_dataset = test_dataset.batch(5)
print(train_dataset)


model.fit(train_dataset, epochs=20)
model.evaluate(test_dataset)