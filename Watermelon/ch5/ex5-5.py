#!/usr/python3
from numpy import *
from sklearn import preprocessing
from sys import argv


def getDataSet():  # 返回的是已经将字符串转化为标量的数据
    with open('ex5-5.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        DataSet = []
        LabelSet = []
        index = 0
        for i in lines:
            LabelSet.append(i.strip().split(',')[-1])
            temp = []
            temp.extend(i.strip().split(',')[:-3])
            temp.append(float(i.strip().split(',')[-3]))
            temp.append(float(i.strip().split(',')[-2]))
            temp.append(i.strip().split(',')[-1])
            DataSet.append(temp)
            index += 1
    retData = zeros((len(DataSet), len(DataSet[0]) - 1))
    for i in range(len(DataSet[0]) - 3):
        retData[:, i] = oneHotEncoder([x[i] for x in DataSet])
    retData[:, 6] = [x[6] for x in DataSet]
    retData[:, 7] = [x[7] for x in DataSet]
    labelArr = oneHotEncoder(LabelSet)
    return retData + 1, labelArr


# 对某一特征（向量）(原矩阵中的某一列向量)进行编码
def oneHotEncoder(dataVec):
    dataLine = [[temp] for temp in dataVec]
    oneL = preprocessing.LabelEncoder()
    oneL.fit(dataLine)
    return oneL.transform(dataLine)


def sigmoid(inX):
    return 1.0 / (1 + exp(-inX))


# 对于sigmoid函数进行求导后整理的形式
def sigmoidDerivative(inX):
    return (sigmoid(inX) * (1 - sigmoid(inX)))


# 非向量化版本
class BP:
    def __init__(self, n_input, n_hidden_layer, n_output, learn_rate, error,
                 n_max_train, value):

        self.n_input = n_input
        self.n_hidden_layer = n_hidden_layer
        self.n_output = n_output
        self.learn_rate = learn_rate
        self.error = error
        self.n_max_train = n_max_train

        # random initialize the weights of each layer
        self.v = random.random((self.n_hidden_layer, self.n_input))
        self.w = random.random((self.n_output, self.n_hidden_layer))

        # initialize the threshold of output layer
        self.theta = random.random(self.n_output)
        # initialize the threshold of hidden layer
        self.gamma = random.random(self.n_hidden_layer)

        # b = the output of  the hidden layer
        self.b = []
        # yo = the output of the hidden layer
        self.yo = []

        self.x = []
        self.y = []
        self.lossAll = []
        self.lossAverage = 0
        self.nRight = 0
        self.value = value

    def printParam(self):
        print('printParam')
        print('---------------')
        print('     v: ', self.v)
        print('     w: ', self.w)
        print('theta0: ', self.theta)
        print('theta1: ', self.gamma)
        print('---------------')

    def init(self, x, y):
        nx = len(x)
        ny = len(y)
        self.x = x
        self.y = y
        self.b = []
        self.yo = []
        self.b = zeros((nx, n_hidden_layer))
        self.yo = zeros((nx, self.n_output))

    def printProgress(self):
        print('yo:', self.yo)

    def calculateLoss(self, y, yo):
        print('!!!y:')
        print(y)
        print('!!!yo:')
        print(yo)
        loss = 0
        for i in range(self.n_output):
            loss += (y[i] - yo[i])**2
        return loss

    def calculateLossAll(self):
        self.lossAll = []
        for i in range(len(self.x)):
            loss = self.calculateLoss([self.y[i]], self.yo[i])
            self.lossAll.append(loss)
        self.lossAverage = sum(self.lossAll) / (len(self.x))

    def calculateOutput(self, x, k):
        for i in range(self.n_hidden_layer):
            self.b[i] = sigmoid(
                float(mat(self.v[i]) * mat(x).T) - self.gamma[i])
        for i in range(self.n_output):
            self.yo[i] = sigmoid(
                float(mat(self.w[i]) * mat(self.b[i]).T) - self.theta[i])

    def printResult(self):
        print('printResult')
        self.calculateLossAll()
        print('lossAll:', self.lossAll)
        print('lossAverage:', self.lossAverage)
        self.nRight = 0
        for k in range(len(self.x)):
            print(self.y[k], '----', self.yo[k])
            self.nRight += 1
            for j in range(self.n_output):
                if (self.yo[k][j] > self.value[j][0]
                        and self.y[k][j] != self.value[j][2]):
                    self.nRight -= 1
                    break
                if (self.yo[k][j] < self.value[j][0]
                        and self.y[k][j] != self.value[j][1]):
                    self.nRight -= 1
                    break
        print('right rate: %d/%d' % (self.nRight, len(self.x)))


class BPStandard(BP):
    def updateParam(self, k):
        g = []
        for i in range(self.n_output):
            g.append(self.yo[k][i] * (1.0 - self.yo[k][i]) *
                     (self.y[k][i] - self.yo[k][i]))
        e = []
        for h in range(self.n_hidden_layer):
            temp = 0
            for j in range(self.n_output):
                temp += self.b[k][h] * (
                    1.0 - self.b[k][h]) * self.w[h][j] * g[j]
            e.append(temp)

        for h in range(self.n_output):
            for j in range(self.n_hidden_layer):
                self.w[h][j] += self.learn_rate * g[h] * self.b[k][j]
        for j in range(self.n_output):
            self.theta[j] -= self.learn_rate * g[j]
        for i in range(self.n_hidden_layer):
            for h in range(self.n_input):
                self.v[i][h] += self.learn_rate * e[i] * self.x[k][h]
        for h in range(self.n_hidden_layer):
            self.gamma[h] -= self.learn_rate * e[h]

    def train(self, x, y):
        print('trian neural networks')
        self.init(x, y)
        self.printParam()
        tag = 0
        loss1 = 0
        print('train begin')
        n_train = 0
        nr = 0
        while 1:
            for k in range(len(x)):
                n_train += 1
                self.calculateOutput(x[k], k)
                self.calculateLossAll()
                loss = self.lossAverage
                if abs(loss1 - loss) < self.error:
                    nr += 1
                    if nr >= 100:
                        break
                else:
                    nr = 0
                    self.updateParam(k)
                if n_train % 10000 == 0:
                    for k in range(len(x)):
                        self.calculateOutput(x[k], k)
                    self.printProgress()
                if n_train > self.n_max_train or nr >= 100:
                    break

        print('train end')
        self.printParam()
        self.printResult()
        print('train count: ', n_train)


# 向量化版本
def bps(x, y, n_hidden_layer, r, error, n_max_train):
    print('standard bp algorithm')
    print('------------------------------------')
    print('init param')
    [xrow, xcol] = x.shape
    [yrow, ycol] = y.shape
    v = random.random((xcol, n_hidden_layer))
    w = random.random((n_hidden_layer, ycol))
    t0 = random.random((1, n_hidden_layer))
    t1 = random.random((1, ycol))
    print('---------- train begins ----------')
    n_train = 0
    tag = 0
    yo = 0
    loss = 0
    while 1:
        for k in range(len(x)):
            b = sigmoid(x.dot(v) - t0)
            yo = sigmoid(b.dot(w) - t1)
            loss = sum((yo - y)**2) / xrow
            if loss < error or n_train > n_max_train:
                tag = 1
                break
            b = b[k]
            b = b.reshape(1, b.size)
            n_train += 1
            g = yo[k] * (1 - yo[k]) * (y[k] - yo[k])
            g = g.reshape(1, g.size)
            w += r * b.T.dot(g)
            t1 -= r * g
            e = b * (1 - b) * g * w.T
            v += r * x[k].reshape(1, x[k].size).T.dot(e)
            t0 -= r * e
            if n_train % 10000 == 0:
                print('train count: ', n_train)
                print(hstack((y, yo)))
        if tag:
            break
    print('---------- train ends ----------')
    print('train count = ', n_train)
    yo = yo.tolist()
    print('---------- learned param: ----------')
    print('---------- result: ----------')
    print(hstack((y, yo)))
    print('loss: ', loss)


if __name__ == '__main__':
    n_hidden_layer = 10
    learn_rate = 0.1
    error = 0.005
    n_max_train = 100000

    x, y = getDataSet()

    z = array(mat(y)).T
    x = array(x)
    bps(x, z, n_hidden_layer, learn_rate, error, n_max_train)
