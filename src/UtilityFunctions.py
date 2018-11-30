import numpy as np;



def accuracy(Y_pred, Y_label):
    Y = np.argmax(Y_pred, axis=0)
    correct = 0
    Y_label = Y_label.flatten()
    for i in range(Y_label.size):
        if Y[i] == Y_label[i]:
            correct += 1
    return correct / Y_label.size * 100

def indices_to_one_hot(data, nb_classes):
    """Convert an iterable of indices to one-hot encoded labels."""
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]