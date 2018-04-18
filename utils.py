'''
frequently used implementations that can be used everywhere :P
'''

import numpy as np
from scipy.misc import imresize
import os
from PIL import Image
from sklearn.preprocessing import normalize
from sklearn import metrics
from os.path import isfile
from vggface import preprocess_input

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
    img1 = _Path2Image(path1)
    img2 = _Path2Image(path2)
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    pair = np.concatenate([img1,img2], axis=0)
    pair = preprocess_input(pair)
    embeddings = model.predict_on_batch(pair)
    dist = _DistanceEuclidean(embeddings[0], embeddings[1])
    return dist
    
def _Path2Image(path):
    im = Image.open(path)
    im = im.resize((224,224))
    im = np.array(im).astype(np.float32)
    return im
    
def _DistanceEuclidean(X, Y):
    X = X.reshape(1, -1)
    Y = Y.reshape(1, -1)
    diff = (normalize(X) - normalize(Y))
    return (diff**2).sum()
    
    
def Inputs2ArrayImage(inputs):
    #convert image path or PIL Image to ndarray Image if needed
    img = None
    if isinstance(inputs, str): # input is a path
        img = Image.open(os.path.expanduser(inputs))
        img = np.array(img)
        # img = misc.imread(os.path.expanduser(input))
    elif isinstance(inputs, Image.Image): # input is a PIL image
        img = np.array(inputs)
    elif isinstance(inputs, (np.ndarray, np.generic)): # input is a numpy array
        img = inputs.astype('uint8')
    else:
        msg = 'unexpected type of input! '
        msg += 'expect str, PIL or ndarray image, '
        msg += 'but got {}'
        raise TypeError(msg.format(type(inputs)))
        
    if len(img.shape)==2:
        img = _Gray2RGB(img)
        
    return img

def _Gray2RGB(img):
    assert len(img.shape)==2
    A = np.expand_dims(img, axis=-1)
    B = np.expand_dims(img, axis=-1)
    C = np.expand_dims(img, axis=-1)
    return np.concatenate([A,B,C],axis=-1)
    
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