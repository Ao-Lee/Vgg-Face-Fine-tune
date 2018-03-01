import os
import numpy as np
from scipy import misc
from sklearn.metrics import pairwise as kernels
from PIL import Image

STRATEGY_ALL = 0
STRATEGY_LARGEST = 1
STRATEGY_CLOSE_TO_CENTER = 2
STRATEGY_PROBABILITY = 3

def SelectLargest(bounding_boxes, img_size, strategy=STRATEGY_ALL):
    assert bounding_boxes.shape[1] ==5
    boxes = bounding_boxes[:,0:4]

    if strategy == STRATEGY_LARGEST or strategy == STRATEGY_ALL:
        areas = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        scores_area = areas/(np.sum(areas))

    if strategy == STRATEGY_PROBABILITY or strategy == STRATEGY_ALL:
        probabilities = bounding_boxes[:,-1]
        scores_prob = probabilities/(np.sum(probabilities))

    if strategy == STRATEGY_CLOSE_TO_CENTER or strategy == STRATEGY_ALL:
        center_img = (img_size/2).reshape(1, -1)
        normalized_center_img = center_img / img_size
        center_boxes = np.vstack([(boxes[:,0]+boxes[:,2])/2, (boxes[:,1]+boxes[:,3])/2]).T
        normalized_center_boxes = center_boxes / img_size
        similarities = kernels.rbf_kernel(X=normalized_center_boxes, Y=normalized_center_img, gamma=5).reshape(-1)
        scores_dist = similarities/(np.sum(similarities))

    if strategy == STRATEGY_LARGEST:
        final_scores = scores_area
    if strategy == STRATEGY_PROBABILITY:
        final_scores = scores_prob
    if strategy == STRATEGY_CLOSE_TO_CENTER:
        final_scores = scores_dist
    if strategy == STRATEGY_ALL:
        final_scores = np.log(scores_area) + np.log(scores_prob) + np.log(scores_dist)
    
    larggest_idx = final_scores.argmax()
    return larggest_idx
    
def Inputs2ArrayImage(input):
    #convert image path or PIL Image to ndarray Image if needed
    img = None
    if isinstance(input,str): # input is a path
        img = misc.imread(os.path.expanduser(input))
    elif isinstance(input, Image.Image): # input is a PIL image
        img = np.array(input)
    elif isinstance(input, (np.ndarray, np.generic)): # input is a numpy array
        img = img.astype('uint8')
    else:
        msg = 'unexpected type of input! '
        msg += 'expect str, PIL or ndarray image, '
        msg += 'but got {}'
        raise TypeError(msg.format(type(input)))
    return img
    
    