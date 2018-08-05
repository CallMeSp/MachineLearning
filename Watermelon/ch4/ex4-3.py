import math
import matplotlib
import matplotlib.pyplot as plt
from numpy import *


def getDataSet():
    with open('ex4-3.txt','r') as f:
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
    threshValue=0.0
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
            threshValue=t
    return newEntropy,threshValue

def chooseBestFeatToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    baseEntropy=getEntropy(dataSet)
    bestInforGain=0.0
    bestFeature=-1
    retBool=False
    retValue=0.0
    for i in range(numFeatures):
        isFloat=False
        threshValue=0.0
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
            isFloat=True
            featList=[example[i] for example in dataSet]
            sortedList=sort(featList)
            Ta=[]
            for k in range(len(sortedList)-1):
                Ta.append(float(sortedList[k]+sortedList[k+1])/2)
            newEntropy,threshValue=getEntropyForFloat(dataSet,i,Ta)
        infoGain=baseEntropy-newEntropy
        if infoGain>bestInforGain:
            bestInforGain=infoGain
            bestFeature=i
            retBool=isFloat
            retValue=threshValue
    return bestFeature,retBool,retValue

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classList.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet)==1:
        return majorityCnt(classList)
    bestFeat,isFloat,threshValue=chooseBestFeatToSplit(dataSet)
    if not isFloat:
        bestFeatLabel=labels[bestFeat]
        myTree={bestFeatLabel:{}}
        del(labels[bestFeat])
        featValues=[example[bestFeat] for example in dataSet]
        uniqueSet=set(featValues)
        for value in uniqueSet:
            subLabels=labels[:]
            myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    else:
        bestFeatLabel=labels[bestFeat]+'<='+str(threshValue)
        myTree={bestFeatLabel:{}}
        subSet0=[];subSet1=[]
        for k in range(len(dataSet)):
            if(dataSet[k][bestFeat]<=threshValue):
                subSet0.append(dataSet[k])
            else:
                subSet1.append(dataSet[k])
        subLabels=labels[:]
        myTree[bestFeatLabel]['是']=createTree(subSet0,subLabels)
        subLabels=labels[:]
        myTree[bestFeatLabel]['否']=createTree(subSet1,subLabels)
    return myTree

if __name__=='__main__':
    dataSet,label=getDataSet()
    featureLabels=['色泽','根蒂','敲声','纹理','脐部','触感','密度','含糖率']
    print(createTree(dataSet,featureLabels))
