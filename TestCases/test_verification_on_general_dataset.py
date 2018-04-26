import numpy as np
from os.path import join, isfile
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from keras import backend as K
import sys
sys.path.append('..')
from vggface import VggFace
from utils import Verify
import cfg

# dir_image = 'E:\\DM\\Faces\\Data\\ARFace\\aligned'
# dir_txt = 'ARFace\\validation.txt'

# dir_image = 'E:\\DM\\Faces\\Data\\LFW\\aligned'
# dir_txt = 'LFW\\validation.txt'

dir_image = 'E:\\DM\\Faces\\Data\\PCD\\aligned'
dir_txt = 'verification_text_files\\PCD\\validation.txt'
K.set_image_dim_ordering('th')


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
    threshold = 1.1
    model = VggFace(cfg.dir_model_v2, is_origin=True)
    labels, dists = GetResult(model)
    labels = (labels=='S').astype(np.int)
    predictions = (dists < threshold).astype(np.int)
    
    auc = metrics.roc_auc_score(np.logical_not(labels), dists)
    acc = metrics.accuracy_score(labels, predictions)
    print('auc is: {}'.format(auc))
    print('accuracy is {}'.format(acc))
