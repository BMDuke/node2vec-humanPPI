'''
Model is implemented based on guide from: 
 > https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/text/word2vec.ipynb#scrollTo=RutaI-Tpev3T

Useful resources: 
 > https://www.tensorflow.org/tensorboard/get_started

'''

import io
# from msilib import sequence
import os
import re
import string
import tqdm
from itertools import chain
from pathlib import Path

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

SEED = 3
AUTOTUNE = tf.data.AUTOTUNE
NUMBER_NEGATIVE_SAMPLES = 5
WINDOW_SIZE = 2
BATCH_SIZE = 1024
BUFFER_SIZE = 10000
SEQUENCE_LENGTH = 10
VOCAB_SIZE = 4096
EMBEDDING_DIM = 128

sentence = "The wide road shimmered in the hot sun"
tokens = list(sentence.lower().split())
# print(len(tokens))

class Word2Vec(tf.keras.Model):

    def __init__(self,  window_size=WINDOW_SIZE, 
                        batch_size=BATCH_SIZE, 
                        buffer_size=BUFFER_SIZE, 
                        sequence_length=SEQUENCE_LENGTH,
                        vocab_size=VOCAB_SIZE, 
                        embedding_dim=EMBEDDING_DIM, 
                        negative_samples=NUMBER_NEGATIVE_SAMPLES, 
                        autotune=AUTOTUNE, 
                        seed=SEED): 

        '''
        Here, we create the word2vec class using the keras
        subclassing API. 
        
        We initialise the parent class and then 
        define two embedding matrices, both of size (vocab_size * 
        embedding_dim). 

        An avenue for experimentation is to vary how we represent the
        embedding. For example could we use a shared embedding?
        Could we combine the embeddings?

        Another avenue for experimentation could be to use the 
        negative_samples argument for the 
        keras.preprocessing.sequence.skipgrams function to see if it
        improves the quality of negatite sampling

        Note: This method of creating negative samples can include words
        which are within the neighbourhood of the target word. This means the
        negative samples arent the context word in question, however they may
        be words which are in the context. 

        The method .create_trainng_data could be improved by using numpy arrays
        instead of python lists to generate the training data
        '''

        super(Word2Vec, self).__init__()

        self.window_size = window_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples
        self.autotune = autotune
        self.seed = seed


    def fetch_dataset(self, save_as, fetch_from):

        path_to_file = tf.keras.utils.get_file(save_as, fetch_from)

        # Display first n lines
        with open(path_to_file) as f:
            lines = f.read().splitlines()
        
        for line in lines[:20]:
            print(line)

        return path_to_file

    def create_text_dataset(self, filepath):

        data = tf.data.TextLineDataset(filepath).filter(lambda x: tf.cast(tf.strings.length(x), bool))

        return data

    def create_vocabulary(self, standardizer, data):

        '''
        standardizer: A funciton to format text
        '''

        vocab_size = self.vocab_size
        sequence_length = self.sequence_length
        batch_size = self.batch_size

        vectorize_layer = layers.TextVectorization(
            standardize=standardizer,
            max_tokens=vocab_size,
            output_mode='int',
            output_sequence_length=sequence_length
        )

        # Create the vocabulary
        vectorize_layer.adapt(data.batch(batch_size))

        # Save the vocabulary
        self.inverse_vocab = vectorize_layer.get_vocabulary()

        # Save the vectorised layer
        self.vectorize_layer = vectorize_layer

        # print(self.inverse_vocab[:20])

    '''
    Node2Vec entry point is going to be at .create_vocabulary
    passing a Dataset object as an argument whcih will be a (n, 1)
    tensor of strings
    '''
    
    def vectorize_text(self, data): 

        batch_size = self.batch_size
        vectorize_layer = self.vectorize_layer
        autotune = self.autotune

        vectorized_data = data.batch(batch_size).prefetch(autotune).map(vectorize_layer).unbatch()

        return vectorized_data    

    def create_positive_skipgrams(self, line, sampling_table=None):

        '''
        Generate positive skipgrams for a given sentence
        '''

        window_size = self.window_size
        vocabulary_size = self.vocab_size
        negative_samples = 0

        positive_skipgrams, _ = tf.keras.preprocessing.sequence.skipgrams(
                                    line,
                                    vocabulary_size=vocabulary_size,
                                    sampling_table=sampling_table,
                                    window_size=window_size,
                                    negative_samples=negative_samples
                                )

        return positive_skipgrams

    def create_negative_skipgrams(self, skipgram):

        '''
        Create a sample of negative skipgrams for a given
        positive skipgram
        '''

        target_word, context_word = skipgram
        vocab_size = self.vocab_size
        negative_samples = self.negative_samples

        context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))

        negative_samples, _, _ = tf.random.log_uniform_candidate_sampler(
            true_classes=context_class,
            num_true=1,
            num_sampled=negative_samples,
            unique=True,
            range_max=vocab_size,
            seed=SEED,
            name='negative_sampling'
        )

        return negative_samples

    def create_example(self, p_skipgram, n_skipgrams):

        negative_samples = self.negative_samples

        target_word, context_word = p_skipgram
        context_class = tf.reshape(tf.constant(context_word, dtype="int64"), (1, 1))

        negative_skipgrams = tf.expand_dims(n_skipgrams, 1)

        # Create a batch of y values
        context = tf.concat([context_class, negative_skipgrams], 0)

        # Create a batch of labels
        label = tf.constant([1] + [0]*negative_samples, dtype="int64")

        # Reshape the tensors to (1,) for target and (NUMBER_NEGATIVE_SAMPLES+1,) for 
        # context and labels
        target = tf.squeeze(target_word)
        context = tf.squeeze(context)
        label = tf.squeeze(label)

        return target, context, label

    def create_training_data(self, data, use_sampling_table=False, use_cache=False, path=None):

        '''
        Given a body of text, create the training data consisting of
        positive and negative skipgrams
        '''

        # If use cache
        # Look for saved dataset
        # If it doesnt exist continue and save the dataset at thge end
        # If it does exist load the dataset and return it

        path = os.path.join(path, 'datasets')

        if (use_cache):
            if (os.path.exists(path)):
                if (os.listdir(path)):
                    dataset = tf.data.experimental.load(path)
                    return dataset

        lines = list(data.as_numpy_iterator())

        for line in lines[:5]:
            print(f"{line} => {[self.inverse_vocab[i] for i in line]}")
            print(len(line))

        targets, contexts, labels = [], [], []

        sampling_table = None

        if (use_sampling_table):

            vocab_size = self.vocab_size
            sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(vocab_size)

        # Iterate over all the lines in the corpus
        print(f"Generating skipgrams")

        for line in tqdm.tqdm(lines):

            # Generate the positive skipgrams
            positive_skipgrams = self.create_positive_skipgrams(line, sampling_table)

            # Iterate over all positive skipgrams and produce 
            # the training examples by generating negative skipgrams 
            # and concatenating them to the positive skipgram
            for p_skipgram in positive_skipgrams:

                n_skipgrams = self.create_negative_skipgrams(p_skipgram)
                target, context, label = self.create_example(p_skipgram, n_skipgrams)

                targets.append(target)
                contexts.append(context)
                labels.append(label)

        buffer_size = self.buffer_size
        batch_size = self.batch_size

        targets = np.array(targets)
        contexts = np.array(contexts)
        labels = np.array(labels)

        dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)

        print()

        if (use_cache):
            tf.data.experimental.save(dataset, path)

        return dataset

    def create_embedding_layers(self):

        '''
        Change this to to a method named 'build' as per recommendations from:
        > https://www.tensorflow.org/guide/keras/custom_layers_and_models#best_practice_deferring_weight_creation_until_the_shape_of_the_inputs_is_known
        '''

        vocab_size = self.vocab_size
        embedding_dim = self.embedding_dim
        negative_samples = self.negative_samples


        self.target_embedding = layers.Embedding(
                            vocab_size,
                            embedding_dim,
                            input_length=1,
                            name='w2v_embedding'
                        )
        
        self.context_embedding = layers.Embedding(
                                    vocab_size,
                                    embedding_dim,
                                    input_length=negative_samples + 1
                                )        

    def call(self, pair):

        target, context = pair

        if (len(target.shape) == 2):
            target = tf.squeeze(target, axis=1)
        
        word_embed = self.target_embedding(target)
        context_embed = self.context_embedding(context)

        dots = tf.einsum('be,bce->bc', word_embed, context_embed)

        return dots

    def compile_model(self): 

        self.compile(
            optimizer='adam',
            loss= tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )



# Testing ground
# w2v = Word2Vec(text_ds, 128)

# print(w2v.vocab)
# print(w2v.vocab_size)
# print(w2v.inverse_vocab)

# print(w2v.vectorize(sentence))

# line = w2v.vectorize(sentence)
# skipgs = w2v.create_positive_skipgrams(line)

# print(skipgs[:10])

# ng = w2v.create_negative_skipgrams(skipgs[0])

# print(f"Target: {skipgs[0][0]}")
# print(f"Context: {skipgs[0][1]}")
# print(ng)
# print([w2v.inverse_vocab[index.numpy()] for index in ng])

# t, c, l = w2v.create_example(skipgs[0], ng)

# print(f"Target: {t}")
# print(f"Context: {c}")
# print(f"Label: {l}")

# t, c, l = w2v.create_training_data([sentence])

# print(f"Targets: {t}")
# print(f"Contexts: {c}")
# print(f"Labels: {l}")


# WORKING EXAMPLE
# def standardiser(input_data):
#     lowercase = tf.strings.lower(input_data)
#     return tf.strings.regex_replace(lowercase,
#                                   '[%s]' % re.escape(string.punctuation), '')

# w2v = Word2Vec()

# save_as = 'shakespeare.txt'
# fetch_from = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'

# file = w2v.fetch_dataset(save_as, fetch_from)

# data = w2v.create_text_dataset(file)

# print(list(data.as_numpy_iterator()))

# w2v.create_vocabulary(standardiser, data)

# vectorized_data = w2v.vectorize_text(data)

# dataset = w2v.create_training_data(vectorized_data, use_sampling_table=True)

# w2v.create_embedding_layers()

# w2v.compile_model()

# LOG_DIR = "Models/logs/tensorboard"

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)

# w2v.fit(dataset, epochs=20, callbacks=[tensorboard_callback])

# os.system(f"tensorboard --logdir {LOG_DIR}")

# print(dataset)