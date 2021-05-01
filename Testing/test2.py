import theano
import sys
import numpy
import collections

floatX=theano.config.floatX

class Classifier(object):
    def __init__(self, n_features):
        # network parameters
        random_seed = 42
        hidden_layer_size = 5
        l2_regularisation = 0.001

        # random number generator
        rng = numpy.random.RandomState(random_seed)

        # setting up variables for the network
        input_vector = theano.tensor.fvector('input_vector')
        target_value = theano.tensor.fscalar('target_value')
        learningrate = theano.tensor.fscalar('learningrate')

        # input->hidden weights
        W_hidden_vals = numpy.asarray(rng.normal(loc=0.0, scale=0.1, size=(n_features, hidden_layer_size)), dtype=floatX)
        W_hidden = theano.shared(W_hidden_vals, 'W_hidden')

        # calculating the hidden layer
        hidden = theano.tensor.dot(input_vector, W_hidden)
        hidden = theano.tensor.nnet.sigmoid(hidden)

        # hidden->output weights
        W_output_vals = numpy.asarray(rng.normal(loc=0.0, scale=0.1, size=(hidden_layer_size, 1)), dtype=floatX)
        W_output = theano.shared(W_output_vals, 'W_output')

        # calculating the predicted value (output)
        predicted_value = theano.tensor.dot(hidden, W_output)
        predicted_value = theano.tensor.nnet.sigmoid(predicted_value)

        # calculating the cost function
        cost = theano.tensor.sqr(predicted_value - target_value).sum()
        cost += l2_regularisation * (theano.tensor.sqr(W_hidden).sum() + theano.tensor.sqr(W_output).sum())

        # calculating gradient descent updates based on the cost function
        params = [W_hidden, W_output]
        gradients = theano.tensor.grad(cost, params)
        updates = [(p, p - (learningrate * g)) for p, g in zip(params, gradients)]

        # defining Theano functions for training and testing the network
        self.train = theano.function([input_vector, target_value, learningrate], [cost, predicted_value], updates=updates, allow_input_downcast=True)
        self.test = theano.function([input_vector, target_value], [cost, predicted_value], allow_input_downcast=True)

def read_dataset(path):
    """Read a dataset, with each line containing a real-valued label and a feature vector"""
    dataset = []
    with open(path, "r") as f:
        for line in f:
            line_parts = line.strip().split()
            label = float(line_parts[0])
            vector = numpy.array([float(line_parts[i]) for i in range(1, len(line_parts))])
            dataset.append((label, vector))
    return dataset


if __name__ == "__main__":
    path_train = sys.argv[1]
    path_test = sys.argv[2]

    # training parameters
    learningrate = 0.1
    epochs = 10

    # reading the datasets
    data_train = read_dataset(path_train)
    data_test = read_dataset(path_test)

    # creating the network
    n_features = len(data_train[0][1])
    classifier = Classifier(n_features)

    # training
    for epoch in range(epochs):
        cost_sum = 0.0
        correct = 0
        for label, vector in data_train:
            cost, predicted_value = classifier.train(vector, label, learningrate)
            cost_sum += cost
            if (label == 1.0 and predicted_value >= 0.5) or (label == 0.0 and predicted_value < 0.5):
                correct += 1
        print("Epoch: " + str(epoch) + ", Training_cost: " + str(cost_sum) + ", Training_accuracy: " + str(float(correct) / len(data_train)))

    # testing
    cost_sum = 0.0
    correct = 0
    for label, vector in data_test:
        cost, predicted_value = classifier.test(vector, label)
        cost_sum += cost
        if (label == 1.0 and predicted_value >= 0.5) or (label == 0.0 and predicted_value < 0.5):
            correct += 1
    print("Test_cost: " + str(cost_sum) + ", Test_accuracy: " + str(float(correct) / len(data_test)))
