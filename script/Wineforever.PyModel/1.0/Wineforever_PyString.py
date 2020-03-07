import numpy as np

def load(filepath):
    f = open(filepath,encoding='utf-8')
    data = f.read()
    return data

def save(data,filepath):
    with open(filepath,'w', encoding='utf-8') as f:
        f.write(data)

def load_from_sheet(filepath):
    data = load(filepath)
    FLAG = [False,False]
    res = ['','']
    Data = {}
    for i in range(len(data)-1):
        if data[i-1]=='［':
            FLAG[0] = True
            res[0] = ''
        elif data[i-1] == '　' and data[i] != '　' and data[i] != '［':
            FLAG[1] = True
            res[1] = ''
        if FLAG[0]:
            res[0] += data[i]
        elif FLAG[1]:
            res[1] += data[i]
        if data[i+1] == '］':
            FLAG[0] = False
            Data[res[0]] = []
        elif data[i+1] == '　' and FLAG[1] and data[i] != '］':
            FLAG[1] = False
            Data[res[0]].append(res[1])
    return Data

def save_to_sheet(data,filepath):
    Data = []
    for key, value in data.items():
        Data.append('［' + key + '］')
        for i in range(len(value)):
            if value[i] == '':
                Data.append('　' + ' ' + '　')
            else:
                Data.append('　' + value[i] + '　')
    Data = ''.join(Data)
    save(Data,filepath)

def Serialization(data,filepath):
    data = np.array(data)
    dims = data.ndim
    shape = data.shape
    Data = []
    Data.append('(')
    for i in range(0,dims):
        Data.append(str(shape[i]))
        if i != dims-1:
            Data.append(',')
    Data.append(')')
    if dims == 1:
        for i in range(0,shape[0]):
            Data.append('[')
            Data.append(str(i))
            Data.append('|')
            Data.append(str(data[i]))
            Data.append(']')
    if dims == 2:
        for i in range(0,shape[0]):
            for j in range(0,shape[1]):
                Data.append('[')
                Data.append(str(i) + ',' + str(j))
                Data.append('|')
                Data.append(str(data[i,j]))
                Data.append(']')
    if dims == 3:
        for i in range(0,shape[0]):
            for j in range(0,shape[1]):
                for k in range(0,shape[2]):
                    Data.append('[')
                    Data.append(str(i) + ',' + str(j) + ',' + str(k))
                    Data.append('|')
                    Data.append(str(data[i,j,k]))
                    Data.append(']')
    if dims == 4:
        for i in range(0,shape[0]):
            for j in range(0,shape[1]):
                for k in range(0,shape[2]):
                    for l in range(0,shape[3]):
                        Data.append('[')
                        Data.append(str(i) + ',' + str(j) + ',' + str(k) + ',' + str(l))
                        Data.append('|')
                        Data.append(str(data[i,j,k,l]))
                        Data.append(']')
    Data = ''.join(Data)
    save(Data,filepath)