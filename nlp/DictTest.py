# -*- coding:utf-8 -*-


def getDict():
    Dict = {}
    with open('dic.txt', 'r', encoding='utf-8') as f:
        i = 0
        lines = f.readlines()
        for line in lines:
            i += 1
            # lineArr = line.strip().split('\t', 1)
            # Dict[lineArr[0]] = lineArr[1].replace('\\n', '')
            print(line)
    return Dict


def unknownProcessing(unkownWord):
    pass


if __name__ == '__main__':
    Dict = getDict()
    inputstr = input("Enter your input: ").lower()
    if (inputstr in Dict):
        print(inputstr, Dict[inputstr])
    else:
        pass
