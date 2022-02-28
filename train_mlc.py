'''
Useful Resources:
    > https://stackoverflow.com/questions/65103526/how-to-save-textvectorization-to-disk-in-
    > https://stackoverflow.com/questions/67914993/how-to-serve-model-in-tensorflow-serving-that-has-custom-function-registered-wit
    
Notees to self: 
    > You will need the text vectorisation layer as a preprocessing layer for the MLC
        so you cant use a work around that disposes the tv layer when loading weights
'''

import os
import tqdm
from pathlib import Path
import numpy as np
import pickle
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers

from Models.Word2Vec import Word2Vec
from Models.Embedding import Embedding
from Models.MultiLabelClassifier import MultiLabelClassifier
from Models.utils.serializer import walk_serializer

path_tf_dataset = 'cache/tf/models/embedding'
path_tv_layer = 'cache/tf/models/embedding/textVectorisation'


# To load the TV layer
#   Define p and q values 
p = 0.5
q = 0.7
#   Construct file path
path = os.path.join(path_tv_layer, f"{p}-{q}.gpickle")
#   Load the pickled config deets
with open(path, 'rb') as pickle_in:
    tvConfig = pickle.load(pickle_in)
#   Define the standardiser funtion and add it to the config
tvConfig['config']['standardize'] = lambda x: x
#   Instntiate a new TV layer from config
tvLayer = layers.TextVectorization.from_config(tvConfig['config'])
#   .adapt it with dummy data
tvLayer.adapt(tf.data.Dataset.from_tensor_slices(['dummy data']))
#   Set the weights from the pickle and set the weights
tvLayer.set_weights(tvConfig['weights'])

# print(tvLayer("10"))
# w2v = tf.keras.models.load_model(
#     path_tf_dataset)

# Load the w2v model, passing in the text vectorisation layer as a custom object
with tf.keras.utils.CustomObjectScope({"walk_serializer": walk_serializer}):
    w2v = tf.keras.models.load_model(
        path_tf_dataset)

print(w2v.summary())
print(w2v.layers)

tv = w2v.get_layer('text_vectorization')
em = w2v.get_layer('w2v_embedding')

# print(tv)
# print(tv(tf.Tensor('0', dtype='string')))
# print(tv(tf.constant('2339')).numpy())
# print(tvConfig['config'])
tv_c = tv.get_config()
tv_w = tv.get_weights()
tv_c['trainable'] = False

tv_c2 = layers.TextVectorization.from_config(tv_c)
tv_c2.adapt(tf.data.Dataset.from_tensor_slices(['dummy data']))
tv_c2.set_weights(tv_w)

print(tv_c2.get_config())
print(tv_c2.get_weights())

em_c = em.get_config()
em_w = em.get_weights()
em_c['trainable'] = False

print(em_c)
# print(len(em_w))

em_c2 = layers.Embedding(
    em_c['input_dim'],
    em_c['output_dim'],
    input_length=em_c['input_length'],
    trainable=False,
    weights=em_w,
)
# em_c2.set_weights(em_w)

print(em_c2)

embedding_block = Embedding(tv_c2, em_c2)
print(embedding_block.trainable_weights)
print(embedding_block.embedding_layer.output_dim)
print(embedding_block.get_output_dim())

# Import the data 
path_to_dataset = 'data/node_classification_vertex_list.csv'

data = pd.read_csv(path_to_dataset)
print(data.head())
print(data.shape)

# X
genes = data["GENE"].to_numpy()
# to_str = np.vectorize(lambda x: str(x))
# to_str = np.vectorize(lambda x: tf.constant(str(x)))
# to_str = np.vectorize(lambda x: np.array(str(x)))
# genes = to_str(genes)
genes = [np.array(str(x)) for x in genes]

print(genes)
# print(genes.shape)

# Y
labels = data.iloc[: , 1:].to_numpy()
# to_tnsr = np.vectorize(lambda x: tf.constant(x, dtype='float32'))
to_tnsr = np.vectorize(lambda x: tf.convert_to_tensor(x, dtype='float32'))
# labels = to_tnsr(labels)

print(labels)
print(labels.shape)

# Labels in english
label_names = data.columns[1:]

print(label_names)
print(label_names.shape)

# Define a function that returns all label names
def get_label_names(prediction):

    matches = []

    for pred, label in zip(prediction, label_names):
        if pred == 1:
            matches.append(label)
    
    return matches

# for i in range(10):
#     print(get_label_names(labels[i, :]))
#     print(labels[i, :])

# print(genes.shape)
print(labels.shape)

print(type(genes))
print(type(genes[1]))

print(type(labels))
print(type(labels[1,:]))

dataset = tf.data.Dataset.from_tensor_slices((genes, labels))
print(next(iter(dataset)))

# for elem in list(dataset.batch(10).as_numpy_iterator()):
#     print(elem)
#     print(elem.shape)

# print(genes.shape)
# print(labels.shape)

# print(type(labels))
# print(type(labels[1,:]))

# Create the MLC
mlc = MultiLabelClassifier(embedding_block, label_names)
mlc.compile_model()
mlc.fit(dataset, epochs=1, verbose=True)

# mlc.build(shape=(,1))
# mlc.summary()

print(tf.expand_dims(genes[0], axis=0))
print(tf.expand_dims(genes[0], axis=0).shape)
# print(tf.reshape(genes[0], shape=(0,1)))

y0 = mlc(genes[0])
# y1 = mlc.predict(tf.expand_dims(genes[1], axis=0))
# y2 = mlc.predict(tf.expand_dims(genes[2], axis=0))
# y3 = mlc.predict(tf.expand_dims(genes[3], axis=0))

print(y0)
# print(y1)
# print(y2)
# print(y3)

for i, j in zip(labels[0], y0):
    print(i, '\t', j.numpy()[0])


# [print(i.shape, i.dtype) for i in mlc.inputs]
# [print(o.shape, o.dtype) for o in mlc.outputs]
# [print(l.name, l.input_shape, l.dtype) for l in mlc.layers]