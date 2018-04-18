'''
conver a standard parisDevTest.txt file from LFW database to standard validation.txt

examples of pairsDevTest:
Abdullah_Gul	13	14
Candice_Bergen	2	Prospero_Pichay	1

examples of validation.txt
Abdullah_Gul\Abdullah_Gul_0013.jpg      Abdullah_Gul\Abdullah_Gul_0014.jpg          S
Candice_Bergen\Candice_Bergen_0013.jpg  Prospero_Pichay\Prospero_Pichay_0014.jpg    D 
'''

import pandas as pd
from os.path import join

path_from = 'pairsDevTest.txt'
path_to = 'validation.txt'

def CreateFilename(name, idx):
    idx = int(idx)
    return name + '_' + '%04d' % idx + '.jpg'
    
def TransforDifferent(inputs):
    assert len(inputs)==4
    name1, idx1, name2, idx2 = inputs
    path1 = join(name1, CreateFilename(name1, idx1))
    path2 = join(name2, CreateFilename(name2, idx2))
    label = 'D'
    return path1, path2, label
    
def TransforSame(inputs):
    assert len(inputs)==3
    name, idx1, idx2 = inputs
    path1 = join(name, CreateFilename(name, idx1))
    path2 = join(name, CreateFilename(name, idx2))
    label = 'S'
    return path1, path2, label
    

    
if __name__=='__main__':
    list_path1 = []
    list_path2 = []
    list_label = []
    with open(path_from) as file:
        for line in file:
            # print(line)
            line = line.split(sep='\t')
            if len(line)==4: #deffierent person
                path1, path2, label = TransforDifferent(line)
            else: # same person
                path1, path2, label = TransforSame(line)
            list_path1.append(path1)
            list_path2.append(path2)
            list_label.append(label)
        
    df = pd.DataFrame([])
    df['img1'] = pd.Series(list_path1)
    df['img2'] = pd.Series(list_path2)
    df['class'] = pd.Series(list_label)
    df.to_csv(path_to, sep='\t', index=False, header=True)
    
    