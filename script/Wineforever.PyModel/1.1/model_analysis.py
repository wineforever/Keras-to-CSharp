import h5py
from Wineforever_PyString import save_to_sheet
from Wineforever_PyBar import Bar
from Wineforever_PyString import Serialization
import numpy as np

model = input("please enter model name(eg:model.h5):")
if(model==''):
    model = 'model.h5'

model = h5py.File('input\\'+model,'r')

tip = input('is show net structure?(y/n):')
if(tip==''):
    tip='y'

if(tip == 'y'):
    for a in model.keys():
        print(' -- %s' % a)
        for b in model[a].keys():
            print(' ------ %s' % b)
            for c in model[a][b].keys():
                print(' ---------- %s' % c)
                try:
                    for d in model[a][b][c].keys():
                        print(' ------------ %s' % d)
                except:print('',end='')

tip = input('is show data shape?(y/n):')
if(tip==''):
    tip='y'

if(tip == 'y'):
    for a in model.keys():
        for b in model[a].keys():
            for c in model[a][b].keys():
                try:
                    for d in model[a][b][c].keys():
                        print('%s %s -------- %s' % (c,d,str(model[a][b][c][d].shape)))
                except:print('',end='')

tip = input('fast read or customize read?(fr/cr):')
if(tip==''):
    tip = 'fr'

if(tip=='fr'):
    print('please wait...')
    keys = model['model_weights'].keys()
    layers_num = len(keys)
    dense_count = 1
    conv_count = 1
    for n in keys:
        if 'dense' in str(n):
            temp = model['model_weights']['dense_' + str(dense_count)]['dense_' + str(dense_count)]
            '''
            for weights
            '''
            data = []
            len_0 = temp['kernel:0'].shape[0]
            len_1 = temp['kernel:0'].shape[1]
            bar = Bar('progress',len_0*len_1)
            for i in range(0,len_0):
                data_ = []
                for j in range(0,len_1):
                    data_.append(temp['kernel:0'][i][j])
                    bar.next()
                data.append(data_)
            Serialization(data, 'Output\\dense_'+str(dense_count) + '_weights.group')
            print('dense_weights_' + str(dense_count) + ' done!')
            '''
            for bias
            '''
            data = []
            len_0 = temp['bias:0'].shape[0]
            bar = Bar('progress',len_0)
            for i in range(0,len_0):
                data.append(temp['bias:0'][i])
                bar.next()
            Serialization(data, 'Output\\dense_' + str(dense_count) + '_bias.group')
            print('dense_bias_' + str(dense_count) + ' done!')
            dense_count = dense_count + 1
        if 'conv2d' in str(n):
            temp = model['model_weights']['conv2d_' + str(conv_count)]['conv2d_' + str(conv_count)]
            '''
            for weights
            '''
            data = []
            len_0 = temp['kernel:0'].shape[0]
            len_1 = temp['kernel:0'].shape[1]
            len_2 = temp['kernel:0'].shape[2]
            len_3 = temp['kernel:0'].shape[3]
            bar = Bar('progress',len_0*len_1*len_2*len_3)
            for i in range(0,len_0):
                data_ = []
                for j in range(0,len_1):
                    data__ = []
                    for k in range(0,len_2):
                        data___ = []
                        for l in range(0,len_3):
                            data___.append(temp['kernel:0'][i][j][k][l])
                            bar.next()
                        data__.append(data___)
                    data_.append(data__)
                data.append(data_)
            Serialization(data, 'Output\\conv_'+str(conv_count) + '_weights.group')
            print('conv_weights_' + str(conv_count) + ' done!')
            '''
            for bias
            '''
            data = []
            len_0 = temp['bias:0'].shape[0]
            bar = Bar('progress',len_0)
            for i in range(0,len_0):
                data.append(temp['bias:0'][i])
                bar.next()
            Serialization(data, 'Output\\conv_' + str(conv_count) + '_bias.group')
            print('conv_bias_' + str(conv_count) + ' done!')
            conv_count = conv_count + 1
    print('done!')

if(tip=='cr'):

    path = ''

    tip = input('please enter first name(eg:model_weights):')
    if(tip==''):
        tip='model_weights'

    dataset = model[tip]
    path += tip
    print('path: %s' % path)

    tip = input('please enter second name(eg:dense_1):')
    if(tip==''):
        tip='dense_1'

    dataset = dataset[tip]
    path += '-->' + tip
    print('path: %s' % path)

    tip = input('please enter third name(eg:dense_1):')
    if(tip==''):
        tip='dense_1'

    dataset = dataset[tip]
    path += '-->' + tip
    print('path: %s' % path)

    tip = input('please enter fourth name(eg:bias:0):')
    if(tip==''):
        tip='bias:0'

    dataset = dataset[tip]
    path += '-->' + tip
    print('path: %s' % path)

    tip = input('is 1 dim,2 dims or more?(eg:1):')
    if(tip==''):
        tip='1'

    if(tip=='1'):
        data = {}
        tip = input('please enter its length:')
        len = int(tip)
        data['bias'] = []
        bar = Bar('progress',len)
        name = input('please enter its name:')
        for i in range(0,len):
            data['bias'].append(str(dataset[i]))
            bar.next()
        save_to_sheet(data, 'Output\\'+name + '.wf')
        print('done!')

    if(tip=='2'):
        data = {}
        tip = input('please enter dim 0 length:')
        len_0 = int(tip)
        tip = input('please enter dim 1 length:')
        len_1 = int(tip)
        bar = Bar('progress',len_1)
        name = input('please enter its name:')
        for i in range(0,len_1):
            data['kernel_' + str(i)] = []
            for j in range(0,len_0):
                data['kernel_' + str(i)].append(str(dataset[j][i]))
            bar.next()
        save_to_sheet(data, 'Output\\'+name + '.wf')
        print('done!')

    if(tip=='3'):
        data = []
        tip = input('please enter dim 0 length:')
        len_0 = int(tip)
        tip = input('please enter dim 1 length:')
        len_1 = int(tip)
        tip = input('please enter dim 2 length:')
        len_2 = int(tip)
        bar = Bar('progress',len_0)
        name = input('please enter its name:')
        for i in range(0,len_0):
            data_ = []
            for j in range(0,len_1):
                data__ = []
                for k in range(0,len_2):
                    data__.append(dataset[i][j][k])
                data_.append(data__)
            data.append(data_)
            bar.next()
        Serialization(data, 'Output\\'+name + '.group')
        print('done!')

    if(tip=='4'):
        data = []
        tip = input('please enter dim 0 length:')
        len_0 = int(tip)
        tip = input('please enter dim 1 length:')
        len_1 = int(tip)
        tip = input('please enter dim 2 length:')
        len_2 = int(tip)
        tip = input('please enter dim 3 length:')
        len_3 = int(tip)
        bar = Bar('progress',len_0)
        name = input('please enter its name:')
        for i in range(0,len_0):
            data_ = []
            for j in range(0,len_1):
                data__ = []
                for k in range(0,len_2):
                    data___ = []
                    for l in range(0,len_3):
                        data___.append(dataset[i][j][k][l])
                    data__.append(data___)
                data_.append(data__)
            data.append(data_)
            bar.next()
        Serialization(data, 'Output\\'+name + '.group')
        print('done!')