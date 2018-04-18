'''
frequently used implementations that can be used everywhere :P
'''
import numpy as np
from scipy.misc import imresize
from sklearn.preprocessing import normalize
from sklearn import metrics
from os.path import isfile

import cfg
from vggface import preprocess_input
from UtilsAlign import Inputs2ArrayImage

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
    
def Verify(path1, path2, model):
    assert(isfile(path1))
    assert(isfile(path2))
    img1 = Inputs2ArrayImage(path1, dtype=np.float32, size=cfg.image_size)
    img2 = Inputs2ArrayImage(path2, dtype=np.float32, size=cfg.image_size)
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    pair = np.concatenate([img1,img2], axis=0)
    pair = preprocess_input(pair)
    embeddings = model.predict_on_batch(pair)
    dist = _DistanceEuclidean(embeddings[0], embeddings[1])
    return dist
        
def _DistanceEuclidean(X, Y):
    X = X.reshape(1, -1)
    Y = Y.reshape(1, -1)
    diff = (normalize(X) - normalize(Y))
    return (diff**2).sum()
    

    
# if u got two imgs with same size, and u wanna show them together in one shot, here is what u got
def MergeImage(img1, img2, how='auto', color=(40,40,40), margin='auto', min_size=600):

    assert how in ['vertical', 'horizontal', 'auto']
    assert img1.shape==img2.shape
    h = img1.shape[0]
    w = img1.shape[1]
    if how == 'auto':
        how = 'horizontal' if h < w else 'vertical'
    color = np.array(color,dtype=np.uint8)
    if margin == 'auto':
        margin = min(h, w)//20

    new_h = h + margin*2 if how=='vertical' else h*2 + margin*3
    new_w = w + margin*2 if how=='horizontal' else w*2 + margin*3
    
    new_img = np.zeros([new_h, new_w, 3], dtype=np.uint8)
    new_img[:,:,:] = color

    new_img[margin:margin+h,margin:margin+w,:] = img1

    if how == 'vertical':
        start = margin*2 + w
        end = margin*2 + w*2
        new_img[margin:margin+h, start:end, :] = img2
    if how == 'horizontal':
        start = margin*2 + h
        end = margin*2 + h*2
        new_img[start:end, margin:margin+w, :] = img2

    size = min(new_w, new_h)
    ratio = 1 if size<= min_size else min_size/size
    new_w = int(new_w*ratio)
    new_h = int(new_h*ratio)
    new_img = imresize(new_img, (new_h, new_w))
    return new_img