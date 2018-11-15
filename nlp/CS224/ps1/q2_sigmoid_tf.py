import tensorflow as tf
import numpy as np


def sigmoid(x):
    x = np.array(x, float)
    with tf.Session() as sess:
        return sess.run(tf.nn.sigmoid(x))


def sigmoid_grad(s):
    with tf.Session() as sess:
        return sess.run(tf.multiply(s, 1 - s))


def test_sigmoid_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    x = np.array([[1, 2], [-1, -2]])
    f = sigmoid(x)
    g = sigmoid_grad(f)
    print('f', f)
    f_ans = np.array([[0.73105858, 0.88079708], [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)
    print('g', g)
    g_ans = np.array([[0.19661193, 0.10499359], [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)
    print("You should verify these results by hand!\n")


if __name__ == "__main__":
    test_sigmoid_basic()