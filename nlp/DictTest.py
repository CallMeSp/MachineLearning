# -*- coding:utf-8 -*-
import re


def getDict():
    Dict = {}
    with open('dic_ec.txt', 'r', encoding='utf-8') as f:
        i = 0
        lines = f.readlines()
        for line in lines:
            i += 1
            lineArr = line.strip().split('\uf8f5', 1)
            Dict[lineArr[0]] = lineArr[1].replace('\uf8f5', '')
    return Dict


def unknownProcessing(unkownWord):

    pass


if __name__ == '__main__':
    x = 'sbsoosks'
    print(re.match(r'(.*)s', x).group())
    # Dict = getDict()
    # inputstr = input("Enter your input: ").lower()
    # if (inputstr in Dict):
    #     print(inputstr, Dict[inputstr])
    # else:
    #     unknownProcessing(inputstr)
