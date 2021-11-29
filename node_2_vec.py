'''
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
'''