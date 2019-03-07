"""This was adapted from rbm.py for a Machine learning class assignment.
"""

from __future__ import print_function

import timeit

import numpy

import theano
import theano.tensor as T
import os

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from logistic_sgd import load_data

from rbm import RBM

def hw3_rbm(learning_rate=0.1, training_epochs=15,
             batch_size=1,n_chains=2,
             n_hidden=3,n_visible=4):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param batch_size: size of a batch used to train the RBM

    :param n_chains: number of parallel Gibbs chains to be used for sampling

    :param n_visible: number of visible nodes

    :param n_hidden: number of hidden nodes

    """

    train_set_x = theano.shared(
                value=numpy.random.random_sample((3,4)),
                name='train_set_x',borrow=True)
    
    test_set_x = theano.shared(
                value=numpy.random.random_sample((3,4)),
                name='test_set_x',borrow=True)

    print('Training set')
    print(train_set_x.get_value(borrow=True))
    print('')
    print('Test set')
    print(test_set_x.get_value(borrow=True))

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    
    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(numpy.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)

    hbias = theano.shared(
                value=numpy.random.random_sample(n_hidden),
                name='hbias',
                borrow=True
            )

    vbias = theano.shared(
                value=numpy.random.random_sample(n_visible),
                name='vbias',
                borrow=True
            )

    # construct the RBM class
    rbm = RBM(input=x, n_visible=n_visible,
              n_hidden=n_hidden, numpy_rng=rng, theano_rng=theano_rng, hbias=hbias, vbias=vbias)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rbm.get_cost_updates(lr=learning_rate,
                                         persistent=persistent_chain, k=15)

    #################################
    #     Training the RBM          #
    #################################

    # start-snippet-5
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    start_time = timeit.default_timer()

    # go through training epochs
    for epoch in range(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in range(n_train_batches):
            mean_cost += [train_rbm(batch_index)]

        print('Training epoch %d, cost is ' % epoch, numpy.mean(mean_cost))

        print('Weights:')
        print(rbm.W.get_value(borrow=True))
        
        print('')

    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - 0.

    print ('Training took %f minutes' % (pretraining_time / 60.))
    # end-snippet-5 start-snippet-6
    #################################
    #     Sampling from the RBM     #
    #################################
    # find out the number of test samples
    number_of_test_samples = test_set_x.get_value(borrow=True).shape[0]

    # pick random test examples, with which to initialize the persistent chain
    test_idx = rng.randint(number_of_test_samples - n_chains)
    persistent_vis_chain = theano.shared(
        numpy.asarray(
            test_set_x.get_value(borrow=True)[test_idx:test_idx + n_chains],
            dtype=theano.config.floatX
        )
    )
    

if __name__ == '__main__':
    hw3_rbm()