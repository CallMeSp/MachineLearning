from numpy import *
from sklearn import preprocessing
def getDataSet():
    with open('ex5-5.txt','r',encoding='utf-8') as f:
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

# 对某一特征进行编码
def oneHotEncoder(dataVec):
    dataLine=[[temp] for temp in dataVec]
    oneL=preprocessing.LabelEncoder()
    oneL.fit(dataLine)
    return oneL.transform(dataLine)

def getOneHotTest():
    dataSet,labels=getDataSet()
    for i in range(len(dataSet[0])-1):
        label0=[x[i] for x in dataSet]
        oneH=preprocessing.OneHotEncoder()
        oneL=preprocessing.LabelEncoder()
        oneL.fit([[temp] for temp in label0])
        label0_lableEncoder=oneL.transform([[temp] for temp in label0])
        oneH.fit([[temp] for temp in label0_lableEncoder])
        lable0_hotEncoder=oneH.transform([[temp] for temp in label0_lableEncoder])
        print(label0_lableEncoder)
        # print(lable0_hotEncoder.toarray())

