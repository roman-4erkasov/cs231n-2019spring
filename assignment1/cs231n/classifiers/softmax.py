from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # https://datascience.stackexchange.com/questions/29735/how-to-apply-the-gradient-of-softmax-in-backprop
    # https://stackoverflow.com/questions/41663874/cs231n-how-to-calculate-gradient-for-softmax-loss-function

    N = X.shape[0] #number of observations
    C = W.shape[1] #number of classes
    D = W.shape[0] #number of dimentions
    
    p = np.zeros([N, C])
    
    #loss computation
    for i in range(N): 
        numerator = np.exp(X[i,y[i]])
        # X=[N,D], W=[D,C] x=[1,D]
        # xW=[1,C] - linear model
        o = X[i].dot(W)
        numerator = np.exp(o[y[i]])
        denominator = np.sum(np.exp(o))
        loss += -np.sum(np.log(numerator/denominator))
    loss/=N
    loss+=.5*reg*np.sum(W*W)
    
    #computing gradient
    for i in range(N):
        scores = np.exp(X[i].dot(W))[None,:]
        scores_correct = scores[0,y[i]]
        scores_sum = scores.sum()
        p=scores/scores_sum
        indicator = (np.arange(C)==y[i])[None,:]
        #dW=[D,C]
        dW += (X[i][:,None].dot(p-indicator))
    dW=dW/N+reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    D = X.shape[1]
    C = W.shape[1]
    ##### loss ######
    #X=[N,D] W=[D,C] o=[N,C]
    o = X.dot(W)
    exp_o = np.exp(o)
    numerator = exp_o[range(N),y]
    denomerator = np.sum(exp_o, axis=1)
    loss = -np.sum(np.log(numerator/denomerator))/N + .5*reg*np.sum(W*W)
    
    ##### grad ######
    # P=[N,C]
    P = exp_o/denomerator[:, None]
    # Y=[N,C] - indicator matrix c==y[i]
    Y = np.array([[int(c==y[i]) for c in range(C)] for i in range(N)])
    # dW=[D,C] W=[D,C]
    dW = X.T.dot(P-Y)/N+reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
