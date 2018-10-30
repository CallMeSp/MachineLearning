import tensorflow as tf
import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def normalizeRows(x):
    pass


def test_normalize_rows():
    print("Testing normalizeRows...")
    x = normalizeRows(np.array([[3.0, 4.0], [1, 2]]))
    print(x)
    ans = np.array([[0.6, 0.8], [0.4472136, 0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print("sucTest")


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    pass


def getNegativeSamples(target, dataset, K):
    """ Samples K indexes which are not the target """

    indices = [None] * K
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target:
            newidx = dataset.sampleTokenIdx()
        indices[k] = newidx
    return indices


def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
                               K=10):
    pass


def skipgram(currentWord,
             C,
             contextWords,
             tokens,
             inputVectors,
             outputVectors,
             dataset,
             word2vecCostAndGradient=softmaxCostAndGradient):
    pass


def cbow(currentWord,
         C,
         contextWords,
         tokens,
         inputVectors,
         outputVectors,
         dataset,
         word2vecCostAndGradient=softmaxCostAndGradient):
    pass


def word2vec_sgd_wrapper(word2vecModel,
                         tokens,
                         wordVectors,
                         dataset,
                         C,
                         word2vecCostAndGradient=softmaxCostAndGradient):
    pass


def test_word2vec():
    """ Interface to the dataset for negative sampling """
    dataset = type('dummy', (), {})()

    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], \
            [tokens[random.randint(0,4)] for i in range(2*C)]

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10, 3))
    dummy_tokens = dict([("a", 0), ("b", 1), ("c", 2), ("d", 3), ("e", 4)])
    print("==== Gradient check for skip-gram ====")
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)
    print("\n==== Gradient check for CBOW      ====")
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, softmaxCostAndGradient),
        dummy_vectors)
    gradcheck_naive(
        lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient),
        dummy_vectors)

    print("\n=== Results ===")
    print(
        skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens,
                 dummy_vectors[:5, :], dummy_vectors[5:, :], dataset))
    print(
        skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5, :],
                 dummy_vectors[5:, :], dataset, negSamplingCostAndGradient))
    print(
        cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5, :],
             dummy_vectors[5:, :], dataset))
    print(
        cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5, :],
             dummy_vectors[5:, :], dataset, negSamplingCostAndGradient))


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()