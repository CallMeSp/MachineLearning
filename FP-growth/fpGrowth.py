class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name=nameValue
        self.count=numOccur
        self.nodeLink=None
        self.parent=parentNode
        self.children={}
    
    def inc(self,numOccur):
        self.count+=numOccur

    def disp(self,ind=1):
        print(' '*ind,self.name,' ',self.count)
        for child in self.children.values():
            child.disp(ind+1)

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict
    
def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children:#check if orderedItems[0] in retTree.children
        inTree.children[items[0]].inc(count) #incrament count
    else:   #add items[0] to inTree.children
        inTree.children[items[0]] = treeNode(items[0], count, inTree)
        if headerTable[items[0]][1] == None: #update header table 
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]])
    if len(items) > 1:#call updateTree() with remaining ordered items
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)
        
def updateHeader(nodeToTest, targetNode):   #this version does not use recursion
    while (nodeToTest.nodeLink != None):    #Do not use recursion to traverse a linked list!
        nodeToTest = nodeToTest.nodeLink
    nodeToTest.nodeLink = targetNode

def createTree(dataSet,minSup=1):
        headerTable={}
        for trans in dataSet:
            for item in trans:
                headerTable[item]=headerTable.get(item,0)+dataSet[trans]
        for k in headerTable.keys():
            if headerTable[k]<minSup:
                del(headerTable[k])
        freqItemSet=set(headerTable.keys())
        if len(freqItemSet)==0:
            return None,None
        for k in headerTable:
            headerTable[k]=[headerTable[k],None]
        retTree=treeNode('Null Set',1,None)
        for tranSet,count in dataSet.items():
            localD={}
            for item in tranSet:
                if item in freqItemSet:
                    localD[item]=headerTable[item][0]
            if len(localD)>0:
                orderItems=[v[0] for v in sorted(localD.items(),key=lambda p:p[1],reverse=True)]
                updateTree(orderItems,retTree,headerTable,count)
        return retTree,headerTable    
    
