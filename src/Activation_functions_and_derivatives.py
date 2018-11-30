import numpy as np;
from src import UtilityFunctions as UF;

no_of_digits = 10;


def tanh(Z):
    '''
    computes tanh activation of Z

    Inputs:
        Z is a numpy.ndarray (n, m)

    Returns:
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.tanh(Z)
    cache = {}
    cache["Z"] = Z
    return A, cache


def tanh_der(dA, cache):
    '''
    computes derivative of tanh activation

    Inputs:
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input
        to the activation layer during forward propagation

    Returns:
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    # We have dA in the input all we need is Z and
    # that we get from that cache which also the part of the input this method

    dZ = dA * (1.0 - np.tanh(cache["Z"]) ** 2)[0];
    return dZ;


def sigmoid(Z):
    '''
    computes sigmoid activation of Z

    Inputs:
        Z is a numpy.ndarray (n, m)

    Returns:
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = 1 / (1 + np.exp(-Z))
    cache = {}
    cache["Z"] = Z
    return A, cache


def sigmoid_der(dA, cache):
    '''
    computes derivative of sigmoid activation

    Inputs:
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input
        to the activation layer during forward propagation

    Returns:
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    # We have dA in the input all we need is Z and
    # that we get from that cache which also the part of the input this method
    A = sigmoid(cache["Z"])[0];
    dZ = dA * (A * (1 - A));
    return dZ;


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def relu(Z):
    '''
    computes relu activation of Z

    Inputs:
        Z is a numpy.ndarray (n, m)

    Returns:
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = np.maximum(0, Z)
    cache = {}
    cache["Z"] = Z
    return A, cache


def relu_der(dA, cache):
    '''
    computes derivative of relu activation

    Inputs:
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input
        to the activation layer during forward propagation

    Returns:
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True)
    Z = cache["Z"]
    dZ[Z < 0] = 0
    return dZ


def linear(Z):
    '''
    computes linear activation of Z
    This function is implemented for completeness

    Inputs:
        Z is a numpy.ndarray (n, m)

    Returns:
        A is activation. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}
    '''
    A = Z;
    cache = {};
    cache["Z"] = Z;
    return A, cache;


def linear_der(dA, cache):
    '''
    computes derivative of linear activation
    This function is implemented for completeness

    Inputs:
        dA is the derivative from subsequent layer. numpy.ndarray (n, m)
        cache is a dictionary with {"Z", Z}, where Z was the input
        to the activation layer during forward propagation

    Returns:
        dZ is the derivative. numpy.ndarray (n,m)
    '''
    dZ = np.array(dA, copy=True);
    return dZ;


def softmax_cross_entropy_loss(Z, Y=np.array([])):
    '''
    Computes the softmax activation of the inputs Z
    Estimates the cross entropy loss

    Inputs:
        Z - numpy.ndarray (n, m)
        Y - numpy.ndarray (1, m) of labels
            when y=[] loss is set to []

    Returns:
        A - numpy.ndarray (n, m) of softmax activations
        cache -  a dictionary to store the activations later used to estimate derivatives
        loss - cost of prediction
    '''
    # Here from Z we find A using softmax approach

    cache = {};
    cache["Z"] = Z;

    A = softmax(Z);
    cache["A"] = A;
    if (Y == np.array([])):
        return A, cache;
    # NO. of sample of
    m = Y.shape[1];

    # This finds loss for individual sample stores the same to 1 * m matrix
    # range(m) varies till and Y acts as the nth dimension each column in A matrix and this works for all other
    # values of Y not equal to ak vanishes
    # one hot representation of labels
    Y_one_hot = UF.indices_to_one_hot(Y.astype(int), no_of_digits);
    A = np.clip(A, 1e-12, 1. - 1e-12)
    log_loss_for_all_samples = -np.sum(Y_one_hot.T * np.log(A))
    # log_loss_for_all_samples = -np.log(A[range(m),int((Y.T)[range(m)])]);
    loss = np.sum(log_loss_for_all_samples) / m;

    return A, cache, loss;


def softmax_cross_entropy_loss_der(Y, cache):
    '''
    Computes the derivative of softmax activation and cross entropy loss

    Inputs:
        Y - numpy.ndarray (1, m) of labels
        cache -  a dictionary with cached activations A of size (n,m)

    Returns:
        dZ - numpy.ndarray (n, m) derivative for the previous layer
    '''
    # finding A from cached value of Z
    # coming up with one hot representation for labels Y
    # subsequently finding dZ

    A = cache["A"];
    Y_one_hot_representation = UF.indices_to_one_hot(Y.astype(int), no_of_digits);
    dZ = A + Y_one_hot_representation.T * (-1);
    return dZ;
