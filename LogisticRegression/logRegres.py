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
    print(x,'---',y)
    ax.plot(x,y.transpose())
    plt.xlabel('X1');plt.ylabel('X2中文');
    plt.show()

if(__name__=='__main__'):
    dataArr,labelMat=loadDataSet()
    weights=gradAscent(dataArr,labelMat)
    print(weights)
    plotBestFit(weights)