import numpy as np
from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def getDataSet():
    with open('3-3.txt','r') as f:
        lines=f.readlines()
        DataSet=zeros((len(lines),3))
        LabelSet=[]
        index=0
        for i in lines:
            LabelSet.append(int(i.strip().split(' ')[-1]))
            # 多加一列常数１方便转换为矩阵相乘
            temp=[];temp.extend(i.strip().split(' ')[:-1]);temp.append(1)
            DataSet[index,:]=temp
            index+=1
    return DataSet,array(LabelSet)

def gradAscent(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)
    alpha=0.001
    maxCycles=10000
    weights=ones((n,1))
    for i in range(maxCycles):
        h=sigmoid(dataMatrix*weights)
        error=(labelMat-h)
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights

def showplt():
    DataSet,LabelSet=getDataSet()
    weights=gradAscent(DataSet,LabelSet)

    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(DataSet[LabelSet==1,0],DataSet[LabelSet==1,1],marker='o')
    ax.scatter(DataSet[LabelSet==0,0],DataSet[LabelSet==0,1],marker='x')
    x=arange(0,1,0.1)
    y=(-weights[2]-weights[0]*x)/weights[1]
    ax.plot(x,y.transpose())
    plt.show()
    
def sigmoid(inX):
    return 1.0/(1+exp(-inX))

if __name__=="__main__":
    
    showplt()