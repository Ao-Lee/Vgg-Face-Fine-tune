import numpy as np
from sklearn.metrics import pairwise as kernels
import os
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

'''
convert an input to PIL image or numpy array image
inputs could be (1)string (2)PIL Image (3) numpy array image
'''    
def Inputs2ArrayImage(inputs, output_type=np.ndarray, dtype='uint8', size=None):
    assert output_type in [np.ndarray, Image.Image]

    img = None
    if isinstance(inputs, str): # input is a path
        img = Image.open(os.path.expanduser(inputs))
    elif isinstance(inputs, Image.Image): # input is a PIL image
        pass
    elif isinstance(inputs, np.ndarray): # input is a numpy array
        img = Image.fromarray(np.uint8(inputs))
    else:
        msg = 'unexpected type of input! '
        msg += 'expect str, PIL or ndarray image, '
        msg += 'but got {}'
        raise TypeError(msg.format(type(inputs)))
        
    if size is not None:
        img = img.resize((size, size))
        
    if output_type==np.ndarray:
        img = np.array(img).astype(dtype)
        if len(img.shape)==2:
            img = _Gray2RGB(img)
        
    return img
    
'''
def Inputs2ArrayImage(inputs):
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
'''
    
def _Gray2RGB(img):
    assert len(img.shape)==2
    R = np.expand_dims(img, axis=-1)
    G = np.expand_dims(img, axis=-1)
    B = np.expand_dims(img, axis=-1)
    return np.concatenate([R,G,B],axis=-1)

    
    