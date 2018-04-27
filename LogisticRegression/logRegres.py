from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('testSet.txt','r')
    for line in fr.readlines():
        lineArr=line.strip().split()
        # 每行前两个值作为x1，x2，另外加一个x0，设为1.0
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1+exp(-inX))

def gradAscent(dataMatIn,classLabels):
    # 将普通矩阵转化为numpy矩阵数据类型
    dataMatrix=mat(dataMatIn)
    # 矩阵的转置，转化为列矩阵
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    alepha=0.001
    maxCycles=500
    weights=ones((n,1))
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alepha*dataMatrix.transpose()*error
    return weights

def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr=array(dataMat)
    n = shape(dataArr)[0]
    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]
    for i in range(n):
        if int(labelMat[i])==1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='blue')
    x=arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    # to make x and y's shape be same ,transpost y
    ax.plot(x,y.transpose())
    plt.xlabel('X1');plt.ylabel('X2');
    plt.show()

def stocGradAscent0(dataArr,classLabels):
    m,n=shape(dataArr)
    alpha=0.01
    weights=ones(n)
    for i in range(m):
        h=sigmoid(sum(dataArr[i]*weights))
        error=classLabels[i]-h
        weights=weights*alpha*error*dataArr[i]
    return weights

def stocGradAscent1(dataArr, classLabels, numIter=150):
    m,n = shape(dataArr)                                                #返回dataMatrix的大小。m为行数,n为列数。
    weights = ones(n)                                                       #参数初始化
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0+j+i)+0.01                                            #降低alpha的大小，每次减小1/(j+i)。
            randIndex = int(random.uniform(0,len(dataIndex)))                #随机选取样本
            h = sigmoid(sum(dataArr[randIndex]*weights))                    #选择随机选取的一个样本，计算h
            error = classLabels[randIndex] - h                                 #计算误差
            weights = weights + alpha * error * dataArr[randIndex]       #更新回归系数
            del(dataIndex[randIndex])                                         #删除已经使用的样本
    return weights 

def classifyVector(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain=open('horseColicTraining.txt')
    frTest=open('horseColicTest.txt')
    trainingSet=[]
    trainingLabels=[]
    for line in frTrain.readlines():
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights=stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount=0
    numTestVec=0.0
    for line in frTest.readlines():
        numTestVec+=1.0
        currLine=line.strip().split('\t')
        lineArr=[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount+=1
    errorRate=float(errorCount)/numTestVec
    print('the error count is : %f'%errorRate)
    return errorRate

def multiTest():
    numTest=10
    errorSum=0.0
    for i in range(numTest):
        errorSum+=colicTest()
    print('after %d iterations the average error rate is %f'%(numTest,errorSum/float(numTest)))

if(__name__=='__main__'):
    # dataArr,labelMat=loadDataSet()
    # weights=stocGradAscent1(array(dataArr),labelMat)
    # print(weights)
    # plotBestFit(weights)
    multiTest()