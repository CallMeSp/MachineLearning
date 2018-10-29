import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def softmax(x):
    x = np.array(x, float)
    with tf.Session() as sess:
        return sess.run(tf.nn.softmax(x))


def test_softmax_basic():
    """
    Some simple tests to get you started.
    Warning: these are not exhaustive.
    """
    print("Running basic tests...")
    test1 = softmax(np.array([1, 2]))
    print('test1', test1)
    ans1 = np.array([0.26894142, 0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001, 1002], [3, 4]]))
    print('test2', test2)
    ans2 = np.array([[0.26894142, 0.73105858], [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001, -1002]]))
    print('test3', test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    print("You should be able to verify these results by hand!\n")


if __name__ == '__main__':
    test_softmax_basic()
