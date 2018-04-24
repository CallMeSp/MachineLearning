# -*- coding: UTF-8 -*-
from numpy import *
import feedparser
import operator
import re

def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        # 集合的并集
        vocabSet=vocabSet|set(document)
    return list(vocabSet)

# 将是否出现作为一个特征：词集模型
def setOfWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print('the word: %s is not in my vocabulary'%word)
    return returnVec

# 将出现次数作为特征：词袋模型
def bagofWords2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
        else:
            print('the word: %s is not in my vocabulary'%word)
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    # 训练样本中的文档数量
    numTrainDocs=len(trainMatrix)
    # 词汇表中单词数量
    numWords=len(trainMatrix[0])
    # 侮辱性文档占全部文档的比重
    pAbusive=sum(trainCategory)/float(numTrainDocs)

    p0Num=ones(numWords)
    p1Num=ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Num+=trainMatrix[i]
            p0Denom+=sum(trainMatrix[i])
    p1Vect=log(p1Num/p1Denom)
    p0Vect=log(p0Num/p0Denom)

    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    p1=sum(vec2Classify*p1Vec)+log(pClass1)
    p0=sum(vec2Classify*p0Vec)+log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def testingNB():
    listPosts,listClasses=loadDataSet()
    myVocabList=createVocabList(listPosts)
    trainMat=[]
    for postinDoc in listPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
    p0V,p1V,pAb=trainNB0(trainMat,listClasses)

    testEntry=['love','my','dalmation']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,' is classified as :',classifyNB(thisDoc,p0V,p1V,pAb))

    testEntry=['stupid','garbage']
    thisDoc=array(setOfWords2Vec(myVocabList,testEntry))
    print(testEntry,' is classified as :',classifyNB(thisDoc,p0V,p1V,pAb))

def textParse(bigString):
    listofTokens=re.findall(r'\w+',bigString)
    return [token.lower() for token in listofTokens if len(token)>2]

def spamTest():
    docList=[];classList=[];fullText=[]
    # 导入并解析文件
    for i in range(1,26):
        # 垃圾邮件
        wordList=textParse(open('email/spam/%d.txt'%i,'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        # 正常邮件
        wordList=textParse(open('email/ham/%d.txt'%i,'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    
    vocabList=createVocabList(docList)
    trainingSet=list(range(50));testSet=[]
    # 随机划分成训练集和测试集
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    trainingMat=[];trainingClasses=[]
    for docIndex in trainingSet:
        trainingMat.append(setOfWords2Vec(vocabList,docList[docIndex]))
        trainingClasses.append(classList[docIndex])
    
    p0v,p1v,pSpam=trainNB0(trainingMat,trainingClasses)

    errorCount=0

    for docIndex in testSet:
        wordVector=setOfWords2Vec(vocabList,docList[docIndex])
        if classifyNB(wordVector,p0v,p1v,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the error rate is ',float(errorCount)/len(testSet))

def calcMostFreq(vocabList,fullText):
    freqDict={}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFrq=sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return sortedFrq[:7]

def localWords(feed1,feed0):
    docList=[];classList=[];fullText=[]
    minLen=min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):

        wordList=textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

        wordList=textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
    vocabList=createVocabList(docList)
    print(vocabList)
    # 去掉最高频的几个单词
    topWords=calcMostFreq(vocabList,fullText)
    for t in topWords:
        # 字典中的key
        if t[0] in vocabList:
            vocabList.remove(t[0])
        
    trainingSet=list(range(2*minLen))
    testSet=[]
    for i in range(5):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainingMat=[]
    trainingClasses=[]
    for docIndex in trainingSet:
        trainingMat.append(bagofWords2Vec(vocabList,docList[docIndex]))
        trainingClasses.append(classList[docIndex])
    p0v,p1v,pSpam=trainNB0(trainingMat,trainingClasses)
    errorCount=0
    for docIndex in testSet:
        wordVect=bagofWords2Vec(vocabList,docList[docIndex])
        if classifyNB(wordVect,p0v,p1v,pSpam)!=classList[docIndex]:
            errorCount+=1
    print('the err rate is ',float(errorCount)/len(testSet))

if(__name__=='__main__'):
    ny=feedparser.parse('http://losangeles.craigslist.org/tfr/index.rss') 
    sf=feedparser.parse('http://newyork.craigslist.org/res/index.rss')
    localWords(ny,sf)