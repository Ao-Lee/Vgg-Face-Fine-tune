import tensorflow as tf
import numpy as np
from .base import detect
from PIL import Image
from .common import Inputs2ArrayImage, SelectLargest
 
# convert a path to traing or testing image
def Convert(inputs, landmark_func, output_size, ec_y):
    F = landmark_func
    img = Inputs2ArrayImage(inputs)
    img = Image.fromarray(np.uint8(img))
    landmark = F(img)
    img = _HorizontalEyes(img, landmark)
    img = _Resize(img, landmark, ec_mc_y=48)
    landmark_new = F(img)
    img = _Crop(img, landmark_new, output_size, ec_y)
    # img = img.convert('L') # convert to gray style
    return np.asarray(img) # convert to ndarray
    
def GetAlignFuncByLandmarks(output_size, ec_y):
    F = GetLargestLandmark()
    return lambda inputs: Convert(inputs, F, output_size, ec_y)

def _GetLargestLandmark_impl(inputs, pnet, rnet, onet):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    img = Inputs2ArrayImage(inputs)
    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, landmarks = detect.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    num_faces = bounding_boxes.shape[0]
    if num_faces == 0:
        print('Unable to align')
        return
    idx = SelectLargest(bounding_boxes, img_size)
    landmarks = [point[idx] for point in landmarks]
    return landmarks
    
def _GetLandmarks_impl(inputs, pnet, rnet, onet):
    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor
    img = Inputs2ArrayImage(inputs)
    bounding_boxes, landmarks = detect.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)
    return bounding_boxes, landmarks
     
def _LandmarkFunc(F):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect.create_mtcnn(sess, None)
    return lambda inputs: F(inputs, pnet, rnet, onet)
    
def GetLargestLandmark():
    return _LandmarkFunc(_GetLargestLandmark_impl)
    
def GetLandmarks():
    return _LandmarkFunc(_GetLandmarks_impl)
    
def _Path2PIL(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')
            
# rotate image so that eyes are set to be horizontal
def _HorizontalEyes(PILImg, pts):
    k = (pts[6]-pts[5]) / (pts[1]-pts[0])
    angle = np.arctan(k)/np.pi*180
    return PILImg.rotate(angle)
    
#set the distance between the midpoint of eyes and the midpoint of mouth to <ec_mc_y>
def _Resize(PILImg, pts, ec_mc_y):
    midpoint_eye_x = (pts[0] + pts[1])/2
    midpoint_eye_y = (pts[5] + pts[6])/2
    midpoint_mouth_x = (pts[3] + pts[4])/2
    midpoint_mouth_y = (pts[8] + pts[9])/2

    distance = np.sqrt((midpoint_mouth_y - midpoint_eye_y)**2 + (midpoint_mouth_x - midpoint_eye_x)**2)
    w = int(PILImg.size[0] / distance * ec_mc_y)
    h = int(PILImg.size[1] / distance * ec_mc_y)
    return PILImg.resize((w, h), Image.BILINEAR)


# crop the image so that the y axis of midpoint of eyes is <ec_y>      
def _Crop(PILImg, pts, output_size, ec_y):
    midpoint_eye_x = (pts[0] + pts[1])/2
    midpoint_eye_y = (pts[5] + pts[6])/2
    size = output_size
    x = midpoint_eye_x - int(size/2)
    y = midpoint_eye_y - ec_y
    return PILImg.crop((x, y, x + size, y + size))