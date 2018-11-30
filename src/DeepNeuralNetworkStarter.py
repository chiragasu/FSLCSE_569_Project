'''
This file implements a multi layer neural network for a multiclass classifier
Aravamuthan Lakshminarayanan
alaksh17@asu.edu
Hemanth Venkateswara
hkdv1@asu.edu
Oct 2018
'''
import numpy as np;
from src import Activation_functions_and_derivatives as AFD;
from src import UtilityFunctions as UF;
from src import load_mnist_Prof as LMP;
import sys, ast;
import time;
from src.Poly import PolyUpdate as PU;
from src.Adam import AdamUpdate as AU;
from src.Nestrov import NestrovUpdate as NU;
from src.RMSProp import RMSpropUpdate as RPU;

no_of_digits = 10;


def initialize_multilayer_weights(net_dims):
    '''
    Initializes the weights of the multilayer network

    Inputs:
        net_dims - tuple of network dimensions

    Returns:
        dictionary of parameters
    '''

    # initialized network parameters
    # We know that W1 dim (n_h,n_in), also
    # W2 dim (n_fin, n_h) so on and  so forth
    # b1 dim (n_h, 1)
    # b2 dim (n_fin, 1)  so on and  so forth
    # from above relationship we can say that, WL = (net_dims[L+1],net_dims[L])
    # Simmilarily we can also say that, bL = (net_dims[L+1],1)
    # we initialize these with with normal random value and use factor 0.001
    # we could have used xavier's initailization but I am going ahead with value 0.001

    np.random.seed(0);
    numLayers = len(net_dims);
    parameters = {};
    momentumParams = {};
    for l in range(numLayers - 1):
        parameters["W" + str(l + 1)] = np.random.randn(net_dims[l + 1], net_dims[l]) * np.sqrt(1 / net_dims[l + 1]);
        parameters["b" + str(l + 1)] = np.random.randn(net_dims[l + 1], 1) * np.sqrt(1 / net_dims[l + 1]);
        momentumParams["vtw" + str(l + 1)] = np.zeros((net_dims[l + 1], net_dims[l]), dtype=float);
        momentumParams["vtb" + str(l + 1)] = np.zeros((net_dims[l + 1], 1), dtype=float);
        momentumParams["mtw" + str(l + 1)] = np.zeros((net_dims[l + 1], net_dims[l]), dtype=float);
        momentumParams["mtb" + str(l + 1)] = np.zeros((net_dims[l + 1], 1), dtype=float);
    return parameters, momentumParams;


def linear_forward(A, W, b):
    '''
    Input A propagates through the layer
    Z = WA + b is the output of this layer.

    Inputs:
        A - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer

    Returns:
        Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        cache - a dictionary containing the inputs A
    '''
    cache = {};
    cache["A"] = A;
    # Simple forward step for Lth layer ZL with inputs A(L-1), W and b
    Z = np.dot(W, A) + b;
    return Z, cache;


def layer_forward(A_prev, W, b, activation):
    '''
    Input A_prev propagates through the layer and the activation

    Inputs:
        A_prev - numpy.ndarray (n,m) the input to the layer
        W - numpy.ndarray (n_out, n) the weights of the layer
        b - numpy.ndarray (n_out, 1) the bias of the layer
        activation - is the string that specifies the activation function

    Returns:
        A = g(Z), where Z = WA + b, where Z is the numpy.ndarray (n_out, m) dimensions
        g is the activation function
        cache - a dictionary containing the cache from the linear and the nonlinear propagation
        to be used for derivative
    '''
    Z, lin_cache = linear_forward(A_prev, W, b);
    if activation == "relu":
        A, act_cache = AFD.relu(Z);
    elif activation == "linear":
        A, act_cache = AFD.linear(Z);

    cache = {};
    cache["lin_cache"] = lin_cache;
    cache["act_cache"] = act_cache;
    return A, cache;


def multi_layer_forward(X, parameters):
    '''
    Forward propgation through the layers of the network

    Inputs:
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}
    Returns:
        AL - numpy.ndarray (c,m)  - outputs of the last fully connected layer before softmax
            where c is number of categories and m is number of samples in the batch
        caches - a dictionary of associated caches of parameters and network inputs
    '''
    L = len(parameters) // 2
    A = X
    caches = []
    for l in range(1, L):  # since there is no W0 and b0
        A, cache = layer_forward(A, parameters["W" + str(l)], parameters["b" + str(l)], "relu")
        caches.append(cache)

    AL, cache = layer_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], "linear")
    caches.append(cache)
    return AL, caches


def linear_backward(dZ, cache, W, b):
    '''
    Backward prpagation through the linear layer

    Inputs:
        dZ - numpy.ndarray (n,m) derivative dL/dz
        cache - a dictionary containing the inputs A, for the linear layer
            where Z = WA + b,
            Z is (n,m); W is (n,p); A is (p,m); b is (n,1)
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    A_prev = cache["A"]
    # If we consider this a general layer this will give us three
    # outputs dA for the layer before, dW and db from the current layer
    dA_prev = np.dot(W.T, dZ);
    dW = np.dot(dZ, cache["A"].T) / A_prev.shape[1];
    db = np.sum(dZ, axis=1, keepdims=True) / A_prev.shape[1];
    return dA_prev, dW, db;


def layer_backward(dA, cache, W, b, activation):
    '''
    Backward propagation through the activation and linear layer

    Inputs:
        dA - numpy.ndarray (n,m) the derivative to the previous layer
        cache - dictionary containing the linear_cache and the activation_cache
        activation - activation of the layer
        W - numpy.ndarray (n,p)
        b - numpy.ndarray (n, 1)

    Returns:
        dA_prev - numpy.ndarray (p,m) the derivative to the previous layer
        dW - numpy.ndarray (n,p) the gradient of W
        db - numpy.ndarray (n, 1) the gradient of b
    '''
    lin_cache = cache["lin_cache"]
    act_cache = cache["act_cache"]

    if activation == "sigmoid":
        dZ = AFD.sigmoid_der(dA, act_cache)
    elif activation == "tanh":
        dZ = AFD.tanh_der(dA, act_cache)
    elif activation == "relu":
        dZ = AFD.relu_der(dA, act_cache)
    elif activation == "linear":
        dZ = AFD.linear_der(dA, act_cache)
    dA_prev, dW, db = linear_backward(dZ, lin_cache, W, b)
    return dA_prev, dW, db


def multi_layer_backward(dAL, caches, parameters):
    '''
    Back propgation through the layers of the network (except softmax cross entropy)
    softmax_cross_entropy can be handled separately

    Inputs:
        dAL - numpy.ndarray (n,m) derivatives from the softmax_cross_entropy layer
        caches - a dictionary of associated caches of parameters and network inputs
        parameters - dictionary of network parameters {"W1":[..],"b1":[..],"W2":[..],"b2":[..]...}

    Returns:
        gradients - dictionary of gradient of network parameters
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
    '''
    L = len(caches)  # with one hidden layer, L = 2
    gradients = {}
    dA = dAL
    activation = "linear"
    for l in reversed(range(1, L + 1)):
        dA, gradients["dW" + str(l)], gradients["db" + str(l)] = \
            layer_backward(dA, caches[l - 1], \
                           parameters["W" + str(l)], parameters["b" + str(l)], \
                           activation)
        activation = "relu"
    return gradients


def classify(X, parameters, lables, onlyPred=False):
    '''
    Network prediction for inputs X

    Inputs:
        X - numpy.ndarray (n,m) with n features and m samples
        parameters - dictionary of network parameters
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
    Returns:
        YPred - numpy.ndarray (1,m) of predictions
    '''
    #
    # Forward propagate X using multi_layer_forward
    # Get predictions using softmax_cross_entropy_loss
    # Estimate the class labels using predictions

    AL = multi_layer_forward(X, parameters)[0];
    if onlyPred:
        Ypred = AFD.softmax_cross_entropy_loss(AL);
        return Ypred;
    Ypred, _, loss = AFD.softmax_cross_entropy_loss(AL, lables);
    return Ypred, loss;


def update_parameters(parameters, gradients, epoch, learning_rate, decay_rate=0.01, descent_optimization_type=0, momentumParams={}):
    '''
    Updates the network parameters with gradient descent

    Inputs:
        parameters - dictionary of network parameters
            {"W1":[..],"b1":[..],"W2":[..],"b2":[..],...}
        gradients - dictionary of gradient of network parameters
            {"dW1":[..],"db1":[..],"dW2":[..],"db2":[..],...}
        epoch - epoch number
        learning_rate - step size for learning
        decay_rate - rate of decay of step size - not necessary - in case you want to use
    '''
    # epoch = 1;
    # alpha = learning_rate * (1 / (1 + decay_rate * epoch));
    alpha = learning_rate;
    L = len(parameters) // 2
    # Once the required slopes of W and b are found the update
    # we update the value W and b and continue till the required iterations are completed


    for l in range(1, L + 1):
        dw_update = gradients["dW" + str(l)];
        db_update = gradients["db" + str(l)];
        if descent_optimization_type == 1:
            dw_update, db_update = PU.polyUpdateParams(momentumParams["mtw" + str(l)], momentumParams["mtb" + str(l)], dw_update, db_update);
            momentumParams["mtw" + str(l)] = dw_update;
            momentumParams["mtw" + str(l)] = db_update;
        elif descent_optimization_type == 2:
            dw_update, db_update = NU.NestrovUpdateParams(momentumParams["mtw" + str(l)], momentumParams["mtb" + str(l)],
                                                       dw_update, db_update);
            momentumParams["mtw" + str(l)] = dw_update;
            momentumParams["mtw" + str(l)] = db_update;
        elif descent_optimization_type == 3:
            dw_update, db_update = AU.adamUpdateParams(momentumParams["mtw" + str(l)],
                                                          momentumParams["mtb" + str(l)],
                                                          dw_update, db_update);
            momentumParams["mtw" + str(l)] = dw_update;
            momentumParams["mtw" + str(l)] = db_update;
        elif descent_optimization_type == 4:
            dw_update, db_update = RPU.rmsPropUpdateParams(momentumParams["mtw" + str(l)],
                                                          momentumParams["mtb" + str(l)],
                                                          dw_update, db_update);
            momentumParams["mtw" + str(l)] = dw_update;
            momentumParams["mtw" + str(l)] = db_update;
        parameters["W" + str(l)] = parameters["W" + str(l)] - alpha * dw_update;
        parameters["b" + str(l)] = parameters["b" + str(l)] - alpha * db_update;
    return parameters, alpha;


def multi_layer_network(X, Y, validation_data, validation_label, net_dims, num_iterations=500, learning_rate=0.2,
                        decay_rate=0.01, batch_size=50, descent_optimization_type=0):
    print("num iter : " + str(num_iterations) + " batch size: " + str(batch_size));

    '''
    Creates the multilayer network and trains the network

    Inputs:
        X - numpy.ndarray (n,m) of training data
        Y - numpy.ndarray (1,m) of training data labels
        net_dims - tuple of layer dimensions
        num_iterations - num of epochs to train
        learning_rate - step size for gradient descent

    Returns:
        costs - list of costs over training
        parameters - dictionary of trained network parameters
    '''
    parameters, momentumparams = initialize_multilayer_weights(net_dims);
    A0 = X;
    costs = [];
    for ii in range(num_iterations):
        print("epoch " + str(ii));
        # Forward Prop
        # This consists for starting from Ao i.e X
        # and evalute till AL
        # keep hold of cache to be used later in backpropagation step
        cost = classify(X, parameters, Y)[1];
        for i in range(0, len(A0.T), batch_size):
            # print("batch " + str(i));
            if i + batch_size >= len(X.T):
                batch_data = X.T[i:].T
                batch_label = Y.T[i:].T
            else:
                batch_data = X.T[i:i + batch_size].T
                batch_label = Y.T[i:i + batch_size].T
            AL, caches = multi_layer_forward(batch_data, parameters);
            AS, cache, loss = AFD.softmax_cross_entropy_loss(AL, batch_label);
            dZ = AFD.softmax_cross_entropy_loss_der(batch_label, cache);
            # find
            # Backward Prop
            # call to softmax cross entropy loss der
            # call to multi_layer_backward to get gradients
            # call to update the parameters
            gradients = multi_layer_backward(dZ, caches, parameters);
            parameters, alpha = update_parameters(parameters, gradients, ii, learning_rate, decay_rate, );
        if ii % 10 == 0:
            costs.append(cost);
        if ii % 10 == 0:
            print("Cost for training at iteration %i is: %.05f, learning rate: %.05f" % (ii, cost, alpha))
    return costs, "", parameters


def main():
    '''
    Trains a multilayer network for MNIST digit classification (all 10 digits)
    To create a network with 1 hidden layer of dimensions 800
    Run the progam as:
        python deepMultiClassNetwork_starter.py "[784,800]"
    The network will have the dimensions [784,800,10]
    784 is the input size of digit images (28pix x 28pix = 784)
    10 is the number of digits

    To create a network with 2 hidden layers of dimensions 800 and 500
    Run the progam as:
        python deepMultiClassNetwork_starter.py "[784,800,500]"
    The network will have the dimensions [784,800,500,10]
    784 is the input size of digit images (28pix x 28pix = 784)
    10 is the number of digits
    '''
    net_dims = ast.literal_eval(sys.argv[1]);
    net_dims.append(10)  # Adding the digits layer with dimensionality = 10
    print("Network dimensions are:" + str(net_dims));

    digit_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    # dummy training data, but next set of data to be used as test data
    validation_data, validation_label, test_data, test_label = \
        LMP.mnist(noTrSamples=1000, noTsSamples=1000, \
                  digit_range=digit_range, \
                  noTrPerClass=100, noTsPerClass=100);

    # real training data, but next set of data to be used as validation data
    train_data_act, train_label_act, validation_data_new, validation_label_new = \
        LMP.mnist(noTrSamples=5000, noTsSamples=1000, \
                  digit_range=digit_range, \
                  noTrPerClass=500, noTsPerClass=100);

    # initialize learning rate and num_iterations
    learning_rate = 0.1;
    num_iterations = 100;

    train_data_act, train_label_act = UF.unison_shuffled_copies(train_data_act.T, train_label_act.T);

    inp = int(input("Enter 1 for comparing Batch Sizes or 2 for Comparing GDO techniques:"));
    costsList = {};
    parametersList = {};

    is_batch_comparision = True;
    if inp == 1:
        is_batch_comparision = True;
        gdo_opt = int(input("Enter the GDO type : "));
        learning_rate = float(input("Learning rate, you want to check for: "));
        # batch_Sizes = [1, 50, 100, 500, 5000];
        batch_Sizes = [500, 100, 5000];

        for x in batch_Sizes:
            tic = time.time();
            costs, _, parameters = multi_layer_network(train_data_act, train_label_act, validation_data,
                                                       validation_label,
                                                       net_dims, \
                                                       num_iterations=num_iterations, learning_rate=learning_rate,
                                                       batch_size=x);
            toc = time.time();
            print("For batch size : " + str(x) + "time taken was :" + str(toc - tic));
            costsList[x] = costs;
            parametersList[x] = parameters;
            UF.getTrainAndValidationAccuracy(train_data_act, train_label_act, validation_data, validation_label,
                                             parameters);

    if inp == 2:
        is_batch_comparision = False;
        batch_Size = int(input("Enter the Batch Size you want to test for : "));
        learning_rate = float(input("Learning rate, you want to check for: "));

        for key in UF.desent_optimzation_map:
            tic = time.time();
            costs, _, parameters = multi_layer_network(train_data_act, train_label_act, validation_data,
                                                       validation_label,
                                                       net_dims, \
                                                       num_iterations=num_iterations, learning_rate=learning_rate,
                                                       descent_optimization_type=key, batch_size=batch_Size);
            print("For GDO : " + UF.desent_optimzation_map[key] + "time taken was :" + str(toc - tic));
            toc = time.time();
            costsList[x] = costs;
            parametersList[x] = parameters;
            UF.getTrainAndValidationAccuracy(train_data_act, train_label_act, validation_data, validation_label,
                                             parameters);

    UF.plotWithCosts(num_iterations, costsList, True, net_dims);


if __name__ == "__main__":
    main()
