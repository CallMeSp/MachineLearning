from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    sqDiffMat=diffMat**2
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    sortedDistIndicies=distances.argsort()
    classcount={}
    for i in range(k):
        vl=labels[sortedDistIndicies[i]]
        classcount[vl]=classcount.get(vl,0)+1
    sortedclasscount = sorted(classcount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedclasscount[0][0]

def file2Matrix(filename):
    fr=open(filename)
    arraylines=fr.readlines()
    numberOfLines=len(arraylines)
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arraylines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(listFromLine[-1])
        index+=1
    return returnMat,classLabelVector


def draw():
    datas, labels = file2Matrix('datingTestSet2.txt')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datas[:, 1], datas[:, 2], 15.0 * array(map(int,labels)), 15.0 * array(map(int,labels)))
    plt.show()


def autoNorm(dataSet):
    minvals=dataSet.min(0)
    maxvals=dataSet.max(0)
    ranges=maxvals-minvals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minvals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minvals

def normalization():
    dataSet,labels=file2Matrix('datingTestSet2.txt')
    normMat,ranges,minvals=autoNorm(dataSet)
    print normMat
    print ranges
    print minvals
    
def datingClassTest():
    hoRatio=0.10
    dataSet,labels=file2Matrix('datingTestSet.txt')
    normMat,ranges,minvals=autoNorm(dataSet)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],labels[numTestVecs:m],3)
        print "the classifier came back with  : %s,the real answer is : %s"%(classifierResult,labels[i])
        if(classifierResult != labels[i]):
                errorCount+=1
    print "errorCount = %d"%(errorCount)
    print "the total error rate is %f"%(errorCount/float(numTestVecs))

def img2Vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels=[]
    trainingFileList=listdir('trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int (fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2Vector('trainingDigits/%s'%fileNameStr)

    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int (fileStr.split('_')[0])
        vectorUnderTest = img2Vector('testDigits/%s'%fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print "the classifier came back with : %d,the real answer is : %d"%(classifierResult,classNumStr)
        if(classifierResult!=classNumStr):errorCount+=1.0
    print "\nthe total number of errors is : %d"%errorCount
    print "\nthe total error rate is : %f "%(errorCount/float(mTest))

