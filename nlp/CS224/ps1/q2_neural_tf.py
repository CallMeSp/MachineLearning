import tensorflow as tf
import numpy as np
from q1_softmax_tf import softmax
from q2_sigmoid_tf import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive
import time


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    the backward propagation for the gradients for all parameters.

    Notice the gradients computed here are different from the gradients in
    the assignment sheet: they are w.r.t. weights, not inputs.

    Arguments:
    X -- M x Dx matrix, where each row is a training example x.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """
    # Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    # ..................................................
    tfW1 = tf.Variable(tf.random_uniform([Dx, H], minval=0, maxval=1))
    tfb1 = tf.Variable(tf.random_uniform([1, H], minval=0, maxval=1))
    tfW2 = tf.Variable(tf.random_uniform([H, Dy], minval=0, maxval=1))
    tfb2 = tf.Variable(tf.random_uniform([1, Dy], minval=0, maxval=1))

    x = tf.placeholder(tf.float32, name="x_input")
    y = tf.placeholder(tf.float32, name="y_input")

    h = tf.nn.sigmoid(tf.add(tf.matmul(x, tfW1), tfb1))
    y_hat_beforesoftmax = tf.add(tf.matmul(h, tfW2), tfb2)
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=y_hat_beforesoftmax))
    trainStep = tf.train.GradientDescentOptimizer(0.001).minimize(cost)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        for i in range(20000):
            sess.run(trainStep, feed_dict={x: np.mat(data), y: np.mat(labels)})
            if i % 100 == 0:
                print(
                    'after %s ,loss=' % i,
                    sess.run(
                        cost, feed_dict={
                            x: np.mat(data),
                            y: np.mat(labels)
                        }))


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])  # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, np.random.randint(0, dimensions[2] - 1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] +
                             (dimensions[1] + 1) * dimensions[2], )
    forward_backward_prop(data, labels, params, dimensions)
    # gradcheck_naive(
    #     lambda params: forward_backward_prop(data, labels, params, dimensions),
    #     params)


if __name__ == "__main__":
    startTime = time.time()
    sanity_check()
    endTime = time.time()
    print("training took %d seconds" % (endTime - startTime))
