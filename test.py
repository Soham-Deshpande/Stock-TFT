import theano
import numpy
import pydot
# x = theano.tensor.fvector('x')
# target = theano.tensor.fscalar('target')
#
# W = theano.shared(numpy.asarray([0.2, 0.7]), 'W')
# y = (x * W).sum()
#
# cost = theano.tensor.sqr(target - y)
# gradients = theano.tensor.grad(cost, [W])
# W_updated = W - (0.1 * gradients[0])
# updates = [(W, W_updated)]
#
# f = theano.function([x, target], y, updates=updates)
#
# for i in range(10):
#     output = f([1.0, 1.0], 20.0)
#     print(output)


x = theano.tensor.fvector('x')
W = theano.shared(numpy.asarray([0.2, 0.7]), 'W')
y = (x * W).sum()

f = theano.function([x], y)
theano.printing.pydotprint(f, outfile="f.png", var_with_name_simple=True)