import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from scipy.misc import imsave
from os.path import join, isfile

import sys
sys.path.append("..")
from AlignDataBase.align.base import detect
from utils import Inputs2ArrayImage, MergeImage

def _GetLandmarks(inputs, pnet, rnet, onet):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    img = Inputs2ArrayImage(inputs)
    bounding_boxes, landmarks = detect.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    return bounding_boxes, landmarks
    
def _GetLandmarkFunc():
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect.create_mtcnn(sess, None)
    return lambda input : _GetLandmarks(input, pnet, rnet, onet)
    
F = _GetLandmarkFunc()

def DrawPoint(draw, x, y):
    r = 1
    draw.ellipse((x-r, y-r, x+r, y+r), fill=(0,0,255,255))
 
def ModifyImg(img):
    bounding_boxes, landmarks = F(img)
    img = Image.fromarray(img)
    B = bounding_boxes[:,:-1]
    
    draw = ImageDraw.Draw(img)
    for b in B:
        shape = ((b[0], b[1]), (b[2], b[3]))
        draw.rectangle(shape, outline='blue')
        
    for m in landmarks.T:
        DrawPoint(draw, m[0], m[5])
        DrawPoint(draw, m[1], m[6])
        
    return np.array(img)
    

if __name__=='__main__':
    root = 'imgs\detection'
    names = ['01.jpg', '02.jpg']

    for name in names:
        path = join(root, name)
        assert(isfile(path))

        img = Inputs2ArrayImage(path)
        img_modified = ModifyImg(img)
        img_merge = MergeImage(img, img_modified)
        
        new_name = '_' + name
        new_path = join(root, new_name)
        imsave(new_path, img_merge)
    
    
    
      