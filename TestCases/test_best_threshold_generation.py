import numpy as np
from os.path import join, isfile
import pandas as pd
from tqdm import tqdm
from sklearn import metrics

import sys
sys.path.append('..')
from vggface import VggFace
from utils import Verify, GetBestThreshold
import cfg

# dir_image = 'E:\\DM\\Faces\\Data\\ARFace\\aligned'
# dir_txt = 'ARFace\\validation.txt'

# dir_image = 'E:\\DM\\Faces\\Data\\LFW\\aligned'
# dir_txt = 'LFW\\validation.txt'

dir_image = 'E:\\DM\\Faces\\Data\\PCD\\aligned'
dir_txt = 'verification_text_files\\PCD\\validation.txt'

def GetDataFrame():
    return pd.read_table(dir_txt, sep='\t')
    
def GetResult(model):
    df = GetDataFrame()
    dists = []
    labels = []
    for idx in tqdm(range(len(df))):
        path1 = join(dir_image, df.loc[idx,'img1'])
        path2 = join(dir_image, df.loc[idx,'img2'])
        if (not isfile(path1)) or (not isfile(path2)):
            continue
        dist = Verify(path1, path2, model)
        dists.append(dist)
        labels.append(df.loc[idx,'class'])
       
    labels = np.array(labels)
    dists = np.array(dists)
    return labels, dists


if __name__ == '__main__':
    model = VggFace(cfg.dir_model_v2, is_origin=True)
    labels, dists = GetResult(model)
    
    labels = (labels=='S').astype(np.int)
    threshold, _ = GetBestThreshold(dists, labels)
    predictions = (dists < threshold).astype(np.int)
    
    auc = metrics.roc_auc_score(np.logical_not(labels), dists)
    acc = metrics.accuracy_score(labels, predictions)
    print('auc is: {}'.format(auc))
    print('threshold is {}'.format(threshold))
    print('accuracy is {}'.format(acc))
