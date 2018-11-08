Dict = {}


def getDict():
    global Dict
    with open('ce（ms-word）.txt', 'r', encoding='gbk') as f:
        i = 0
        lines = f.readlines()
        for line in lines:
            i += 1
            lineArr = line.split(',', 1)
            key = lineArr[0]
            value = lineArr[1].replace('\n', '')
            Dict[key] = value


if __name__ == '__main__':
    getDict()
    testStr = '南京市长江大桥'
    lenth = len(testStr)
    curStart = 0
    curEnd = 1
    resultArr = []
    while curStart < lenth:
        while testStr[curStart:curEnd] in Dict:
            if curEnd < lenth:
                curEnd += 1
            else:
                resultArr.append(testStr[curStart:curEnd])
                break

        if testStr[curStart:curEnd] in resultArr:
            break
        else:
            resultArr.append(testStr[curStart:curEnd - 1])
            curStart = curEnd - 1

    print(resultArr)