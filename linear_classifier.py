# machine learning imports
from mygrad.nnet.layers import dense
from mygrad.nnet.activations import softmax, relu
from mygrad import Tensor
from mygrad.math import log

# data analysis imports
import nltk
from collections import Counter

# utilities
import numpy as np
import glob
import io

class sentiment_classifier():

    def __init__(self, K, W, b):

        """
        Parameters
        ----------
        K : int
            Hyperparameter, the number of neurons in the first layer

        W : mygrad.Tensor
            Model weights

        b : mygrad.Tensor
            Y-intercept (linear classifier)

        """

        self.W = Tensor( he_normal( (input_dims, K) ))
        self.b = Tensor( np.zeros( (K,), dtype = W.dtype ))
        pass

    def he_normal(shape):
        
        """
        Given the desired shape of array, draws random values
        from a scaled-Gaussian distribution.
        Used to initialize weights randomly.

        Parameters
        ----------
        shape : tuple, int

        Returns
        -------
        numpy.ndarray

        """

        N = shape[0]
        scale = 1 / np.sqrt(2*N)

        return np.random.randn(*shape)*scale

    def sgd(param, rate):
        
        """
        Performs a gradient-descent update on param.

        Parameters
        ----------
        param : mygrad.Tensor
            Parameter to be updated
        
        rate : float
            Step size used in the update

        """

        param.data -= rate*param.grad
        return None

    def compute_accuracy(model_out, labels):
        
        """ 
        Computes the mean accuracy, given the model's predictions
        and the correct labels.

        Parameters
        ----------
        model_out : numpy.ndarray, shape = (N, K)
            The predicted class scores/probabilities for each label
        
        labels : numpy.ndarray, shape = (N, K)
            The correct one-hot encoded labels for the data

        
        Returns
        -------
        float
            The mean classification accuracy of the N samples.
        """

        # how often was the prediction the same as the correct label?
        correct = np.argmax(model_out, axis=1) == np.argmax(labels, axis=1)
        
        return np.mean(correct)

    def cross_entropy_loss(p_pred, p_true):
        
        """
        Computes the mean cross-entropy, ie how dissimilar two distributions are.
        A large cross-entropy (very dissimilar) is bad - not what we want.
        A small cross-entropy (very similar) is good - this is what we want!

        Parameters
        ----------
        p_pred : mygrad.Tensor, shape (N, K)
            N predicted distributions, each over K classes

        p_true : mygrad.Tensor, shape (N, K)
            N correct distributions, each over K classes

        Returns
        -------
        mygrad.Tensor, shape=()
            The mean cross entropy, a scalar
        """

        N = p_pred.shape[0]
        p_logq = (p_true) * log(p_pred)

        return (-1/N) * p_logq.sum()

    def train(learning_rate, iterations):
        learn = []
        accuracy = []

        for i in range(iterations):
            out = relu( dense(
        pass
    
    def parse_file():
    
        # hard coded for testing: generalize later
        with io.open('imdb.vocab','r',encoding='utf8') as f:
            text = f.read()

            corpus = text.encode('ascii', 'ignore')
            bag_of_words = str(corpus)


        bag_of_words = bag_of_words.split('\\n')[100:]
    
        return bag_of_words


    def get_descriptor(text):
        
        bag_of_words = parse_file() # hard-coded for testing, generalize later
    
        word_counts = Counter()
        base_desciptor = np.zeros(len(bag_of_words))
        tokens = nltk.word_tokenize(text)
            
        for token in tokens:
            if token in bag_of_words:
                word_counts[tokens.index(token)] += 1

        for k in word_counts:
            base_desciptor[k] = word_counts[k]
        
        return base_desciptor

    def load_text_files(dirt):
    
        file_paths = glob.glob(dirt +'*.txt')
        documents = list()
        
        for file_path in file_paths:
            with io.open('train/unsup/0_0.txt','r',encoding='utf8') as f:
                unicode_data = f.read()

                document = str(unicode_data.encode('ascii', 'ignore'))
                documents.append(document)
                
        return documents

    def initial_test():
        print(len(load_text_files("train/unsup/")))
        pass

    



