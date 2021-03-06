'''
Model is implemented based on paper:
@misc{grover2016node2vec,
    title={node2vec: Scalable Feature Learning for Networks}, 
    author={Aditya Grover and Jure Leskovec},
    year={2016},
    eprint={1607.00653},
    archivePrefix={arXiv},
    primaryClass={cs.SI}
Code inspiration:
    https://github.com/eliorc/node2vec/blob/master/node2vec/node2vec.py#L122
    https://github.com/aditya-grover/node2vec
    https://www.tensorflow.org/tutorials/text/word2vec
    https://towardsdatascience.com/node2vec-embeddings-for-graph-data-32a866340fef

}

Assumptions:
    > Graph is unweighted: the PPI database only indicates the presence
        or absence of an interaction. it does not provide any more information 
        about the type of interaction. Therefore w_vx = 1
    > We dont actually need to code BFS or DFS. The policies in the random walk 
        procedure is characteristic of them
        

Starting points:
    > Build a test graph to experiment on


The Model:
    > G: Graph consists of V, E & W
    > V: Vertex list
    > E: Edge list
    > W: Weight matrix (w_vx = 1 for all w in W)
    > d: This is the number of dimensions for the feature representation
    > The embedding function uses a single hidden layer
    > k: This is the context size which is the number of co-occuring nodes to include
            in the sample when training

Intuition:
    > This is based on the word2vec embedding method which creates word
        embeddings based on related words
    > The nodes in our graph, in total, represent the vocabulary we have to embed.
    > The graph (nodes and edges) represents the corpus which contains the 
        information about how nodes are related to each other
    > Our training data are the walks we take through the graph. This is equivalent
        to a sliding window in the text
    > Our training data have the form [u (X), N(u) (Y)] where u is the starting node
        and N(u) is the neighbourhood. 
    > K is the context which works as follows. We have have N(u). Perhaps it has 10 nodes
        in there. The length of the walk was 12 so it contains some repeats. The context, k, 
        is a constant shared between all training examples. If k=5, 5 of the 10 nodes will 
        be selected to make up Y. Now X=u, Y=randChoice(N(u), 5, repeat=False)
        !! this is incorrect          v
        infact X=u Y=randChoice(N(u), 1, repeat=False) for all nodes in N(u)
        That is, an example is [target_node, context_node_i]. 
        See definition of a skip-gram for further clarification
    > d is the dimension of the hidden layer. This is the number of features that you are
        learning
    > The input to the model is X=u. This is represented as a one-hot vector of size 
        #nodes. For this project this is 4382 nodes
    > The hidden layer has dimension d. The dimension of the input to hidden matrix
        is 4382 nodes * d hidden units.
    > The hidden layer connects to a softmax output layer of size #nodes. There is one softmax 
        output for each node in the vocabulary. These represent the neighbours of u. Their job
        is to predict whether the node they represent is in the neighbourhood of u. 
    > This means you have to train #node * d * 2 parameters which is a lot. This is why they 
        introduced negative sampling. 
    > Negative sampling reduces the training burden. In the naive approach ALL parameters are 
        updated after exposure to an example. In negative sampling, all the softmax classifiers
        in the output layer that predicted 1 have their weights trained, along with a SAMPLE
        of the softmax classifiers that output 0. This trains some but not all of the classifiers
        which dont think the node they represent is in the inputs neighbourhood. This 
        significantly reduces the number of parameters that need to be trained. 
    > NEXT
    > Once we have learned the node embeddings, we can dispose of the hidden -> output weights
        as they were only needed to train the input -> hidden weights. 
    > We now use the input -> hidden weights as the embedding layer and we can take advantage of
        the information stored in the latent representation to learn models for other tasks. 
        The task we will do, is use the input to hidden to create the embedding, then feed this
        to an architecture to perform multi-label classification based on gene pathways


Tests to write:
    > .preprocess_weights() :   second_order_probs are valid probabilities (sum==1, all are >=0)
                                first_order_probs (weights in d_graph) are valid probabilities (sum==1, all are >=0)
                                integrity of d_graph is maintained after applying the function
    > .generate_walks()     :   ensure sample number == num_walks
                                ensure length of one sample == walk_length
                                ensure every item, with index i, in the sample, is in the 
                                    neighborhood of i-1, for all indices >=1
                                ensure neighbors are selected with the correct transition
                                    probabilities
'''
from pickletools import optimize
import random
import os
import tqdm
import pickle

import networkx as nx

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers

from .Word2Vec import Word2Vec 
from .utils.serializer import walk_serializer

class Node2Vec:

    def __init__(self, graph: nx.Graph, dims: int = 128, num_walks: int = 10, 
                    walk_length: int = 80, p: float = 1, q: float = 1, epochs: int = 20):

        '''
        The Node2Vec class defines object which encapsultes the methods
        required to embed nodes in a latent feature space. 

        Inputs:
        > graph:        Networkx graph instance. Contains the nodes and edge list.
        > dims:         Int. The dimension of the feature space
        > num_walks:    Int. The number of walks per node
        > walk_length:  Int. The length of the walk per node
        > p:            Float. The return parameter (see model definition)
        > q:            Float. The in-out parameter (see model definition)
        '''

        # Insert a predicate to check if is DiGraph here 


        self.graph = graph
        self.dims = dims
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = p
        self.q = q

        self.epochs = epochs

        # Create graph G' to contain transition probabilities
        # G' has the same nodes as G, but edges are weighted by transition 
        # probabilities. Because probabilities based on second degree paths
        # are inherently directed, we use a DiGraph
        self.d_graph = graph.to_directed()

        print(f'\nNew Node2Vec instance created: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges')

    def preprocess_weights(self, use_cache: bool = False, path: str = None):
        # Create G' by calculating the transition probabilities
        # using p & q values. 
        '''
        Steps:
        1. For every node, U, in G
        2.  Get Neighbours, N_1
        3.  For every neighbour, N, in N_1
        4.      Get neighbours N_2:
        5.      For every neighbour, N', in N_2
        6.          weight = 1/p if N' == U
        7.          weight = 1 if N' in U.neighbours()
        8.          weight = 1/q if neither above
        9.      self.d_graph.add_edge()

        Example data structure
        graph: {
            edges: {
                (0, 1): {
                    second_degree_probs: {
                        2: 1.87,
                        3: 0.156,
                        4: 0.89,
                        0: 1.00,
                    }
                }
            }
        }

        Notes: 
        >   Its wasteful to duplicate graph -> d_graph, should change in place
        '''

        if (use_cache):
            success = self.load_graph(path)
            if(success):
                print('Matching graph loaded from cache')
                return 

        graph = self.graph
        d_graph = self.d_graph

        nodes = graph.nodes

        print(f'Preprocessing edge weights for {len(self.graph.nodes)} nodes')

        for source in tqdm.tqdm(nodes):

            first_degree_neighbors = [n for n in d_graph.neighbors(source)]

            # Normalise first degree transition probabilities
            normalising_constant = sum([d_graph[source][neighbor].get('weight', 1.) 
                                            for neighbor in first_degree_neighbors])

            for neighbor in first_degree_neighbors:
                normalised_prob = d_graph[source][neighbor].get('weight', 1.) / normalising_constant
                d_graph[source][neighbor]['weight'] = normalised_prob

            for neighbor in first_degree_neighbors:

                second_degree_neighbors = [n for n in graph.neighbors(neighbor)]

                unnormalised_transition_probabilities = {}

                # print(f'Source: {source}')  
                # print(f'Neighbor: {neighbor}')            

                for destination in second_degree_neighbors:

                    weight = d_graph[neighbor][destination].get('weight', 1)
                    weight = float(weight)
                    
                    # print(f'Destination: {destination}')    

                    # Calculate the modified weights
                    if destination == source:
                        # Backwards probability
                        transition_prob = weight * (1. / self.p)
                        # print(f"Back to source. Weight: {weight} transition {transition_prob}")
                    elif destination in first_degree_neighbors:
                        # BFS-like probability
                        transition_prob = weight * 1
                        # print(f"BFS. Weight: {weight} transition {transition_prob}")
                    else:
                        # DFS-like probability
                        transition_prob = weight * (1. / self.q)
                        # print(f"DFS. Weight: {weight} transition {transition_prob}")

                    unnormalised_transition_probabilities[destination] = transition_prob

                # Normalise second degree transition probabilities
                normalised_transition_probabilities = {}
                normalising_constant = sum(unnormalised_transition_probabilities.values())

                for key, value in unnormalised_transition_probabilities.items():
                    normalised_transition_probabilities[key] = value / normalising_constant

                key = 'second_order_probs'
                d_graph[source][neighbor][key] = normalised_transition_probabilities

        
        # Save this object 
        if (use_cache): 
            self.save_graph(path)

        print()

    def generate_walks(self, num_walks: int = None, walk_length: int = None):
        '''
        Simulate random walks. Can be given specific instructions
        for the number of walks and the walk length

        Input:
        > num_walks:    Int. The number of walks per node
        > walk_length:  Int. The length of the walk per node

        '''
        # Set up
        graph = self.d_graph
        if not num_walks:
            num_walks = self.num_walks
        if not walk_length:
            walk_length = self.walk_length

        walks = []

        nodes = list(graph.nodes)

        # Generate walks

        print(f'Generating {num_walks} random walks for {len(self.graph.nodes)} nodes')

        for walk in tqdm.tqdm(range(num_walks)):

            # print(f"Walk iteration: {walk}")
            random.shuffle(nodes)

            for source in nodes:

                sample = [source]

                # Select a node from the neighborhood of source
                neighbors = [n for n in graph.neighbors(source)]
                probabilities = [graph[source][n]['weight'] for n in neighbors]
                neighbor = random.choices(neighbors, weights=probabilities, k=1)[0]
                sample.append(neighbor)

                # Generate samples
                start_node = source             # Node 0 in 2nd order markovian
                current_node = neighbor         # Node 1 in 2nd order markovian
                next_node = None                # Node 2 in 2nd order markovian. We are sampling for this one.

                while len(sample) < walk_length:

                    second_order_probs = graph[start_node][current_node]['second_order_probs']

                    neighbors, probabilities = [], []

                    for key, value in second_order_probs.items():

                        neighbors.append(key)
                        probabilities.append(value)

                    if len(neighbors) == 0:
                        continue

                    next_node = random.choices(neighbors, weights=probabilities, k=1)[0]

                    sample.append(next_node)

                    start_node = current_node
                    current_node = next_node
                    next_node = None
                
                walks.append(sample)

        print()

        return walks

    def stringify(self, walks):

        return [' '.join([str(word) for word in line]) for line in walks]


        
                    
                    


    
    def fit(self, walks, epochs=None, use_cache=False, path=None):
        '''
        Input:
        > Take the context window size as input here

        Steps:
        > Take the generated walks as input [(num_nodes * num_walks) * walk_length]
        > Define targets, contexts and labels as empty lists
        > For each walk, generate skipgram pairs
        > For each target in skip gram pairs
        >   Create negative sample

        # Learn the node embeddings via a lookup table using
        # stochastic gradient descent

        '''

        vocab = self.d_graph.nodes
        vocab_size = len(vocab) + 10 # for any unknown, empty or padded tokens that could get added
        embedding_dim = self.dims
        seed = 3
        autotune = tf.data.AUTOTUNE
        negative_samples = 5
        window_size = 5
        batch_size = 1024
        buffer_size = 10000
        sequence_length = self.walk_length
        epochs = epochs if epochs else self.epochs
        
        stringified_walks = self.stringify(walks)

        dataset = tf.data.Dataset.from_tensor_slices(stringified_walks)

        # Instantiate Word2Vec model
        w2v = Word2Vec( window_size=window_size, 
                        batch_size=batch_size, 
                        buffer_size=buffer_size, 
                        sequence_length=sequence_length,
                        vocab_size=vocab_size, 
                        embedding_dim=embedding_dim, 
                        negative_samples=negative_samples, 
                        autotune=seed, 
                        seed=autotune)

        w2v.create_vocabulary(walk_serializer, dataset) 

        vectorised_data = w2v.vectorize_text(dataset)

        dataset = w2v.create_training_data(vectorised_data, use_cache=use_cache, path=path)

        w2v.create_embedding_layers()

        w2v.compile_model()

        path_logs = os.path.join(path, 'logs')
        path_checkpoint = os.path.join(path, 'checkpoints/checkpoint')
        path_model = os.path.join(path, 'models/embedding')
        path_tv_layer = os.path.join(path, 'models/embedding/textVectorisation')

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=path_logs)
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                filepath=path_checkpoint,
                                save_weights_only=False, # This is False to try and save TextVectorisation layer
                                verbose=1)

        w2v.fit(dataset, epochs=epochs, callbacks=[tensorboard_callback, checkpoint_callback])

        if (use_cache): 
            self.save_tv_layer(w2v.vectorize_layer, path_tv_layer) 

        w2v.save(path_model)

        w2v.summary()

        # embedding_layer = w2v.get_layer('w2v_embedding')

        # print(embedding_layer.get_weights())

        os.system(f"tensorboard --logdir {path_logs}")

        # print(list(dataset.as_numpy_iterator()))
        # print(self.d_graph.nodes)
        # print(w2v.inverse_vocab)
        # print(len(w2v.inverse_vocab))

    def save_graph(self, path): 
        '''
        Pickles the d_graph to economise compute for 
        preprocessing weights
        '''

        # graph_hash = hash(str(list(self.graph.edges)))

        filename = f"{path}/{self.p}-{self.q}.gpickle"

        # nx.write_gpickle(self, filename) ## Note: may need to provide graph

        with open(filename, 'wb') as pickle_out:
            pickle.dump(self, pickle_out)

        return filename



    def load_graph(self, path):

        # graph_hash = hash(str(list(self.graph.edges)))

        filename = f"{path}/{self.p}-{self.q}.gpickle"

        if (os.path.exists(filename)): 

            print("LOADING GRAPH")
            # self = nx.read_gpickle(filename)
            with open(filename, 'rb') as pickle_in:
                n2v_in = pickle.load(pickle_in)
                self.graph = n2v_in.graph
                self.d_graph = n2v_in.d_graph
            print(self.graph)
            print(self.d_graph)

            return filename
        
        else:

            return None

    def save_tv_layer(self, layer, filepath):

        '''
        Pickle the text vectorisation layer so that it can 
        be recreated when we load the trained model weights
        '''

        config = layer.get_config()
        weights = layer.get_weights()

        config['standardize'] = None

        print(config)
        print(weights)


        filename = f"{filepath}/{self.p}-{self.q}.gpickle"

        with open(filename, 'wb') as pickle_out:
            pickle.dump({
                "config": config,
                "weights": weights
            }, pickle_out)

        print('PICKLED TV LAYER CONFIG & WEIGHTS')
        
        
    
    def embed_nodes(self):
        # Return a N x D array of node embeddings
        pass
    
    def save_embeddings(self):
        # Persist the node embeddings
        pass
    
    def save_model(self):
        # Persist the model
        pass
    

