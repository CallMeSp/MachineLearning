Dict = {}


def getDict():
    global Dict
    with open('dic_ec.txt', 'r', encoding='utf8') as f:
        i = 0
        lines = f.readlines()
        for line in lines:
            i += 1
            lineArr = line.strip().split('\uf8f5', 1)
            key = lineArr[0]
            value = lineArr[1].replace('\uf8f5', '')
            Dict[key] = value


def unknownProcessing(unknownword):
    # 处理名词情况
    print('processing...')
    if len(unknownword) > 3:
        if unknownword[-3:] == 'ves':
            rawword = unknownword[:-3] + 'f'
            if rawword in Dict:
                return rawword
            rawword = unknownword[:-3] + 'fe'
            if rawword in Dict:
                return rawword
        elif unknownword[-3:] == 'ies':
            rawword = unknownword[:-3] + 'y'
            if rawword in Dict:
                return rawword
    if unknownword[-2:] == 'es':
        rawword = unknownword[:-2]
        if rawword in Dict:
            return rawword
    if unknownword[-1] == 's':
        rawword = unknownword[:-1]
        if rawword in Dict:
            return rawword
    # 动词情况
    # 1.第三人称单数上述规则已经包括了
    # 2.现在进行时
    if len(unknownword) > 5:
        if unknownword[-3:] == 'ing' and unknownword[-4] == unknownword[-5]:
            rawword = unknownword[:-4]
            if rawword in Dict:
                return rawword
    if unknownword[-4:] == 'ying':
        rawword = unknownword[:-4] + 'ie'
        if rawword in Dict:
            return rawword
    if unknownword[:-3] == 'ing':
        rawword = unknownword[:-3]
        if rawword in Dict:
            return rawword
    # 3.过去式过去分词
    if len(unknownword) > 4:
        if unknownword[-2:] == 'ed' and unknownword[-3] == unknownword[-4]:
            rawword = unknownword[:-3]
            if rawword in Dict:
                return rawword
    if unknownword[-3:] == 'ied':
        rawword = unknownword[:-3] + 'y'
        if rawword in Dict:
            return rawword
    if unknownword[-2:] == 'ed':
        rawword = unknownword[:-2]
        if rawword in Dict:
            return rawword
        rawword = unknownword[:-1]
        if rawword in Dict:
            return rawword
    return None


if __name__ == '__main__':
    getDict()
    inputstr = input('Enter your input:')
    if inputstr in Dict:
        print(inputstr, ':', Dict[inputstr])
    else:
        raw = unknownProcessing(inputstr)
        if raw:
            print(raw, ':', Dict[raw])
        else:
            print('No answer')