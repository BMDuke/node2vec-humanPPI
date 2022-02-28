import os
import tqdm
from pathlib import Path
import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

'''
RESOURCES:
> https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
> https://towardsdatascience.com/multi-label-image-classification-in-tensorflow-2-0-7d4cf8a4bc72

Things to do:
> Import SK learn and generate a sample mlc dataset to test on with make_multilabel_classification() 
    funciton 
'''

class MultiLabelClassifier(tf.keras.Model):

    def __init__(self, embedding, labels): 

        '''
        This class will 
        '''

        super(MultiLabelClassifier, self).__init__()
        
        embedding_dim = embedding.get_output_dim()
        output_dim = len(labels)

        self.embedding = embedding
        self.fc_1 = layers.Dense(
            embedding_dim, 
            activation='leaky_relu',
            use_bias=False)
        self.fc_2 = layers.Dense(
            output_dim,
            activation='sigmoid',
        )
       


    def call(self, input):

        x = self.embedding(input)
        # print(x, '\n')
        x = self.fc_1(x)
        # print(x, '\n')
        x = self.fc_2(x)
        # print(x, '\n')
        # print(tf.reshape(x, [1,50]), '\n')

        # return x
        # return tf.reshape(x, [1,50])
        return tf.reshape(x, [50,1])

    def compile_model(self): 

        self.compile(
            optimizer='adam',
            loss= tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
