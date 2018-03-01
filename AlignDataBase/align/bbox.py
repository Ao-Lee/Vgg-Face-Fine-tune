import tensorflow as tf
import numpy as np
from scipy import misc
from .base import detect
from .common import Inputs2ArrayImage, SelectLargest

def _LoadAndAlign(input, pnet, rnet, onet, image_size, margin):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    img = Inputs2ArrayImage(input)
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = detect.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    num_faces = bounding_boxes.shape[0]
    if num_faces == 0:
        print('Unable to align')
        return
    #det = bounding_boxes[0]
    idx = SelectLargest(bounding_boxes, img_size)
    det = bounding_boxes[idx, 0:4]
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0]-margin/2, 0)
    bb[1] = np.maximum(det[1]-margin/2, 0)
    bb[2] = np.minimum(det[2]+margin/2, img_size[1])
    bb[3] = np.minimum(det[3]+margin/2, img_size[0])
    cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
    aligned_images = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
    return  aligned_images

def GetAlignFuncByBoundingBox(output_size=160, margin=44):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect.create_mtcnn(sess, None)
    return lambda input : _LoadAndAlign(input, pnet, rnet, onet, output_size, margin)
    

