import numpy as np
from os.path import join, isfile
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn import metrics
from vggface import VggFace, preprocess_input

threshold = 0.4
#dir_image = 'E:\\DM\\ARFace\\aligned'
#dir_txt = 'E:\\DM\\ARFace\\validation.txt'

dir_image = 'E:\\DM\\VGG-Face\\aligned'
dir_txt = 'E:\DM\VGG-Face\\validation.txt'


model = VggFace()

def GetDataFrame():
    return pd.read_table(dir_txt, sep='\t')
    
def Path2Image(path):
    im = Image.open(path)
    im = im.resize((224,224))
    im = np.array(im).astype(np.float32)
    return im
    
def DistanceEuclidean(X, Y):
    X = X.reshape(1, -1)
    Y = Y.reshape(1, -1)
    diff = (normalize(X) - normalize(Y))
    return (diff**2).sum()
    
    
def GetResult():
    df = GetDataFrame()
    dists = []
    labels = []
    for idx in tqdm(range(len(df))):
        path1 = join(dir_image, df.loc[idx,'img1'])
        path2 = join(dir_image, df.loc[idx,'img2'])
        if (not isfile(path1)) or (not isfile(path2)):
            continue
        img1 = Path2Image(path1)
        img2 = Path2Image(path2)
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        pair = np.concatenate([img1,img2], axis=0)
        pair = preprocess_input(pair)
        embeddings = model.predict_on_batch(pair)
        dist = DistanceEuclidean(embeddings[0], embeddings[1])
        dists.append(dist)
        labels.append(df.loc[idx,'class'])
       
    labels = np.array(labels)
    dists = np.array(dists)
    return labels, dists

def GetBestThreshold(distance, label):
    def Cost(fpr, tpr):
        return (1-tpr) + fpr

    label_reverse = np.logical_not(label)
    fpr, tpr, thresholds = metrics.roc_curve(label_reverse, distance)
    
    cost = np.inf
    threshold = None
    for idx in range(len(thresholds)):
        current_cost = Cost(fpr[idx], tpr[idx])
        if current_cost < cost:
            cost = current_cost
            threshold = thresholds[idx]
    return threshold, cost
    
if __name__ == '__main__':
    labels, dists = GetResult()
    labels = (labels=='S').astype(np.int)
    threshold, _ = GetBestThreshold(dists, labels)
    predictions = (dists < threshold).astype(np.int)
    
    auc = metrics.roc_auc_score(np.logical_not(labels), dists)
    acc = metrics.accuracy_score(labels, predictions)
    print('auc is: {}'.format(auc))
    print('threshold is {}'.format(threshold))
    print('accuracy is {}'.format(acc))
    

    
