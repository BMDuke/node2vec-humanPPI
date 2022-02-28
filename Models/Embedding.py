'''
This is a subclass of the Layer API.
This will contain the trained TextVectorisation layer
and the Embedding layer from word2vec model. 
This can the be passed as a block to the MLC. 
'''

import tensorflow as tf

class Embedding(tf.keras.layers.Layer):

    def __init__(self, text_vectorisation_layer, embedding_layer):

        super(Embedding, self).__init__()

        self.text_vectorisation_layer = text_vectorisation_layer
        self.embedding_layer = embedding_layer

    def call(self, input):

        x = self.text_vectorisation_layer(input)
        x = self.embedding_layer(x)

        return x
    
    def get_output_dim(self):

        return self.embedding_layer.output_dim