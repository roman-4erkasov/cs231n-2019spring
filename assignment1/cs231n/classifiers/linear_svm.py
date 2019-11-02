from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(num_train):
        # X=[N,D], X[i]=[1,D], W=[D, C]
        # scores = [1, C]
        scores = X[i].dot(W)
        # y=[N,1]
        # field(y[i]) = C
        # correct_class_score=[1,1]
        correct_class_score = scores[y[i]]
        indicator_sum = 0
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1
            if margin > 0:
                indicator_sum += 1
                dW[:, j] += X[i]
        dW[:,y[i]] -= indicator_sum*X[i]
    dW/=num_train
    dW+=reg*W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW


# def svm_loss_vectorized(W, X, y, reg, loss_clip=1e10, grad_clip=1e10):
def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape, dtype=np.float128) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    N = X.shape[0]
    # X=[N,D] W=[D,C] scores=[N,C]
    scores = X.dot(W).astype(dtype=np.float128)
    # y=[N,1] field(y)=C^N, C=classes
    # correct_class_score=[N,1]
    # correct_class_score - cntains score of correct classes
    correct_class_scores = scores[np.arange(N), y].astype(dtype=np.float128)

    margins = np.maximum(scores-correct_class_scores.reshape(N,1)+1.,0.)
    margins[np.arange(N),y] = 0
    # print("!!!", )
    loss = np.mean(np.sum(margins, axis=1), dtype=np.float128)
    loss+=.5*reg*np.sum(W*W)

    # w_norm = np.linalg.norm(W, ord=2)
    # if w_norm>loss_clip:
    #     w_norm = loss_clip
    # loss+=.5*reg*loss_clip
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dscores = np.zeros_like(scores)
    dscores[margins>0] = 1
    dscores[np.arange(N),y] -= np.sum(dscores, axis=1)
    dW = X.astype(dtype=np.float128).T.dot(dscores)/N+reg*W.astype(dtype=np.float128)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
