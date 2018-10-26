import re
def getDict():
    Dict = {}
    with open('dic_ec.txt', 'r', encoding='utf-8') as f:
        i =0
        lines = f.readlines()
        for line in lines:
            i += 1
            lineArr = line.strip().split('\uf8f5', 1)
            Dict[lineArr[0]] = lineArr[1].replace('\uf8f5', '')
    return Dict


def unknownProcessing(unkownWord):
    # 先处理特殊，然后再处理类似y->ies这种固定规则变化
    raw=''
    if re.match(r'(.*)ies$',unkownWord):
        raw=unkownWord[:-3]+'y'
    elif re.match(r'(.*)es$',unkownWord):
        raw=unkownWord[:-2]
    return raw


if __name__ == '__main__':
    print(unknownProcessing('fsafsafiedes'))