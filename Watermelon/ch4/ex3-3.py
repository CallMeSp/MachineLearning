from numpy import *
import math
import matplotlib
import matplotlib.pyplot as plt
def getDataSet():
    with open('ex3-3.txt','r') as f:
        lines=f.readlines()
        DataSet=[]
        LabelSet=[]
        index=0
        for i in lines:
            LabelSet.append(i.strip().split(',')[-1])
            temp=[];temp.extend(i.strip().split(',')[:-3]);temp.append(float(i.strip().split(',')[-3]));temp.append(float(i.strip().split(',')[-2]));temp.append(i.strip().split(',')[-1])
            DataSet.append(temp)
            index+=1
    return DataSet,array(LabelSet)

def getEntropy(dataSet):
    nums=len(dataSet)
    labelCounts={}
    for featvect in dataSet:
        curLabel=featvect[-1]
        if curLabel not in labelCounts.keys():
            labelCounts[curLabel]=0
        labelCounts[curLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/nums
        shannonEnt-=prob*math.log(prob,2)
    return shannonEnt

# 按照某属性的某个值划分，并去掉该属性
def splitDataSet(dataSet,axis,value):
    retDataSet=[]
    for featVec in dataSet:
        if featVec[axis]==value:
            temp=featVec[:axis]
            temp.extend(featVec[axis+1:])
            retDataSet.append(temp)
    return retDataSet

def getEntropyForFloat(dataSet,axis,Ta):
    newEntropy=inf
    for t in Ta:
        D0=[];D1=[]
        for i in range(len(dataSet)):
            if  dataSet[i][axis]<=t:
                D0.append(dataSet[i])
            else:
                D1.append(dataSet[i])
        tempEntropy=float(len(D0))/len(dataSet)*getEntropy(D0)+float(len(D1))/len(dataSet)*getEntropy(D1)
        if tempEntropy<newEntropy:
            newEntropy=tempEntropy
    return newEntropy
    
def chooseBestFeatToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=getEntropy(dataSet)
    bestInforGain=0.0
    bestFeature=-1
    for i in range(numFeatures):
        newEntropy=0.0
        #　区分离散值和连续值
        if isinstance(dataSet[0][i],str):
            featList=[example[i] for example in dataSet]
            uniqueVals=set(featList)
            for value in uniqueVals:
                subDataSet=splitDataSet(dataSet,i,value)
                prob=len(subDataSet)/float(len(dataSet))
                newEntropy+=prob*getEntropy(subDataSet)
        else:
            featList=[example[i] for example in dataSet]
            sortedList=sort(featList)
            Ta=[]
            for k in range(len(sortedList)-1):
                Ta.append(float(sortedList[k]+sortedList[k+1])/2)
            newEntropy=getEntropyForFloat(dataSet,i,Ta)
        infoGain=baseEntropy-newEntropy
        if infoGain>bestInforGain:
            bestInforGain=infoGain
            bestFeature=i
    return bestFeature

if __name__=='__main__':
    dataset,label=getDataSet()
    

    print(chooseBestFeatToSplit(dataset))