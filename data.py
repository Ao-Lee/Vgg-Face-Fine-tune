'''
Note: All images should be aligend first
'''
import numpy as np
import os
from os import listdir
from os.path import join, exists, isdir
from PIL import Image
import matplotlib.pyplot as plt
from abc import ABCMeta,abstractmethod

from vggface import preprocess_input
import cfg


def _CountFiles(root):
    count = 0
    subfolders = [join(root, subfolder) for subfolder in listdir(root)]
    mask = [isdir(subfolder) for subfolder in subfolders]
    count = len(subfolders) - np.sum(mask)
    subfolders = np.array(subfolders)[mask]

    for subfolder in subfolders:
        count += _CountFiles(subfolder)
    return count
       
class DataSetReader:
    __metaclass__ = ABCMeta

    def __init__(self, dir_images):
        self.root = dir_images
        self.data_size = _CountFiles(self.root)
        
    @abstractmethod
    def GetTriplet(self):
        pass
      
    def GetDataSize(self):
        return self.data_size
    
class PCDReader(DataSetReader):
    def __init__(self, dir_images):
        super().__init__(dir_images)
        paths = os.listdir(self.root)
        paths = [join(self.root, path) for path in paths]
        self.num_class = len(paths)
        self.num_per_person = [len(os.listdir(path)) for path in paths]
                       
    @staticmethod
    def CreateName(root, personID, imageID):
        p = '%03d'%personID
        i = '%02d'%imageID + '.jpg'
        return join(root, p, i)
          
    def GetTriplet(self):
        path_anchor, path_pos, path_neg = '', '', ''
        idx_person_pos = np.random.randint(low=0, high=self.num_class)
        L = self.num_per_person
        
        while not exists(path_anchor):
            idx_img_anchor = np.random.randint(low=0, high=L[idx_person_pos])
            path_anchor = PCDReader.CreateName(self.root, idx_person_pos, idx_img_anchor)
            
            
        idx_img_pos=-1
        while (not exists(path_pos)) or  idx_img_pos==idx_img_anchor:
            idx_img_pos = np.random.randint(low=0, high=L[idx_person_pos])
            path_pos = PCDReader.CreateName(self.root, idx_person_pos, idx_img_pos)
        
        idx_person_neg=-1
        while (not exists(path_neg)) or idx_person_neg==idx_person_pos:
            idx_person_neg = np.random.randint(low=0, high=self.num_class)
            idx_img_neg = np.random.randint(low=0, high=L[idx_person_neg])
            path_neg = PCDReader.CreateName(self.root, idx_person_neg, idx_img_neg)
    
        return path_anchor, path_pos, path_neg
       
class PEALReader(DataSetReader):
    def __init__(self, dir_images):
        super().__init__(dir_images)
        paths = os.listdir(self.root)
        self.dir_sex_list=[path[:1] for path in paths]
        paths = [join(self.root, path) for path in paths]
        self.dir=paths
        self.num_class = len(paths)
        self.num_per_person = [len(os.listdir(path)) for path in paths]
        
    @staticmethod
    def CreateName(root, dir, imageID):
        files=[file for file in os.listdir(dir)]
        return (dir+'\\'+files[imageID])
        
    def GetTriplet(self):
        path_anchor, path_pos, path_neg = '', '', ''
        idx_sex = 'M' if np.random.random() > 0.5 else 'F'
        dir_sex=''
        while(idx_sex!=dir_sex):
            idx_person_pos = np.random.randint(low=0, high=self.num_class)
            dir=self.dir[idx_person_pos]
            dir_sex=self.dir_sex_list[idx_person_pos]
        L = self.num_per_person
        
        for file in os.listdir(dir):
            if file[2:8]=='NORMAL':
                path_anchor=(dir+'\\'+file)
        
        while not exists(path_anchor):
            idx_img_anchor = np.random.randint(low=0, high=L[idx_person_pos])
            path_anchor = PEALReader.CreateName(self.root, dir, idx_img_anchor)
            
            
        while (not exists(path_pos)) or  path_anchor==path_pos:
            idx_img_pos = np.random.randint(low=0, high=L[idx_person_pos])
            path_pos = PEALReader.CreateName(self.root, dir, idx_img_pos)
        
        idx_person_neg=-1
        while (not exists(path_neg)) or idx_person_neg==idx_person_pos or self.dir_sex_list[idx_person_neg]!=idx_sex:
            idx_person_neg = np.random.randint(low=0, high=self.num_class)
            idx_img_neg = np.random.randint(low=0, high=L[idx_person_neg])
            path_neg = PEALReader.CreateName(self.root, self.dir[idx_person_neg], idx_img_neg)
    
        return path_anchor, path_pos, path_neg
        
class ARFaceReader(DataSetReader):

    @staticmethod
    def CreateName(root, sex, personID, imageID):
        name = sex + '-' + '%03d'%personID + '-' + '%02d'%imageID
        name = name + '.bmp'
        return join(root, name)
        
    def GetTriplet(self):
        path_anchor, path_pos, path_neg = '', '', ''
        idx_sex = 'M' if np.random.random() > 0.5 else 'W'
        
        while not exists(path_anchor):
            idx_person_pos = np.random.randint(low=1, high=51)
            path_anchor = ARFaceReader.CreateName(self.root, idx_sex, idx_person_pos, 1)
            
        while not exists(path_pos):
            idx_img_pos = np.random.randint(low=2, high=27)
            path_pos = ARFaceReader.CreateName(self.root, idx_sex, idx_person_pos, idx_img_pos)
        
        idx_person_neg=0
        while (not exists(path_neg)) or idx_person_neg==idx_person_pos:
            idx_person_neg = np.random.randint(low=1, high=51)
            idx_img_neg = np.random.randint(low=1, high=27)
            path_neg = ARFaceReader.CreateName(self.root, idx_sex, idx_person_neg, idx_img_neg)
    
        return path_anchor, path_pos, path_neg
        
class LFWReader(DataSetReader):
    def __init__(self, dir_images):
        super().__init__(dir_images)
        self.list_classes = os.listdir(self.root)
        self.not_single = [c for c in self.list_classes if len(listdir(join(self.root, c)))>1]
        self.list_classes_idx = range(len(self.list_classes))
        self.not_single_idx = range(len(self.not_single))
        
        self.weights_not_single = [len(listdir(join(self.root, c))) for c in self.not_single]
        self.weights_not_single = np.array(self.weights_not_single)
        self.weights_not_single = self.weights_not_single / np.sum(self.weights_not_single)
        
        self.weights = [len(listdir(join(self.root, c))) for c in self.list_classes]
        self.weights = np.array(self.weights)
        self.weights = self.weights / np.sum(self.weights)
         
    def GetTriplet(self):
        # positive and anchor classes are selected from folders where have more than two pictures
        idx_class_pos = np.random.choice(self.not_single_idx, 1 ,p=self.weights_not_single)[0]
        name_pos = self.not_single[idx_class_pos]
        dir_pos = join(self.root, name_pos)
        [idx_img_anchor, idx_img_pos]= np.random.choice(range(len(listdir(dir_pos))), 2, replace=False)
        
        # negative classes are selected from all folders
        while True:
            idx_class_neg = np.random.choice(self.list_classes_idx, 1, p=self.weights)[0]
            if idx_class_neg != idx_class_pos:
                break
        name_neg = self.list_classes[idx_class_neg]
        dir_neg = join(self.root, name_neg)
        idx_img_neg = np.random.choice(range(len(listdir(dir_neg))), 1)[0]
        
        # picture names starts from 1, not 0
        idx_img_anchor += 1
        idx_img_pos += 1
        idx_img_neg += 1
        
        path_anchor = join(dir_pos, name_pos+'_'+'%04d' % idx_img_anchor + '.jpg')
        path_pos = join(dir_pos, name_pos+'_'+'%04d' % idx_img_pos + '.jpg')
        path_neg = join(dir_neg, name_neg+'_'+'%04d' % idx_img_neg + '.jpg')

        return path_anchor, path_pos, path_neg

'''
mix a list of readers together
usage:
    reader_PCD = PCDReader(dir_images=cfg.path_PCD)
    reader_AR = ARFaceReader(dir_images=cfg.path_AR)
    reader_LFW = LFWReader(dir_images=cfg.path_LFW)
    reader = MixedReader([reader_PCD, reader_AR, reader_LFW])
'''
class MixedReader(DataSetReader):
    def __init__(self, list_readers):
        self.readers = list_readers
        
    def GetTriplet(self):
        sizes = np.array([reader.GetDataSize() for reader in self.readers])
        p = sizes/np.sum(sizes)
        idx = np.random.choice(range(len(self.readers)), size=1 ,p=p)[0]
        path_anchor, path_pos, path_neg = self.readers[idx].GetTriplet()
        return path_anchor, path_pos, path_neg
        
    def GetDataSize(self):
        return np.sum([reader.GetDataSize() for reader in self.readers])
        
def _ReadAndResize(filepath):
    im = Image.open((filepath)).convert('RGB')
    im = im.resize((cfg.image_size, cfg.image_size))
    return np.array(im, dtype="float32")

def _Flip(im_array):
    if np.random.uniform(0, 1) > 0.7:
        im_array = np.fliplr(im_array)
    return im_array
    
# create triplet example from LFW dataset
def TripletGenerator(reader):
    while True:
        list_pos = []
        list_anchor = []
        list_neg = []

        for _ in range(cfg.batch_size):
            path_anchor, path_pos, path_neg = reader.GetTriplet()
            img_anchor = _Flip(_ReadAndResize(path_anchor))
            img_pos = _Flip(_ReadAndResize(path_pos))
            img_neg = _Flip(_ReadAndResize(path_neg))
            list_pos.append(img_pos)
            list_anchor.append(img_anchor)
            list_neg.append(img_neg)

        A = preprocess_input(np.array(list_anchor))
        P = preprocess_input(np.array(list_pos))
        N = preprocess_input(np.array(list_neg))
        label = None
        
        yield ({'anchor_input': A, 'positive_input': P, 'negative_input': N}, label)        

def _ShowImg(img):
    plt.figure()
    plt.imshow(img.astype('uint8'))
    plt.show()
    plt.close()
    
def _TestTripletGenerator(reader):  
    gen = TripletGenerator(reader)
    data = next(gen)
    imgs_anchor = data[0]['anchor_input']
    imgs_pos = data[0]['positive_input']
    imgs_neg = data[0]['negative_input']
    print(imgs_anchor.shape)
    print(imgs_pos.shape)
    print(imgs_neg.shape)
    imgs_anchor = imgs_anchor.transpose([0,2,3,1])
    imgs_pos = imgs_pos.transpose([0,2,3,1])
    imgs_neg = imgs_neg.transpose([0,2,3,1])
    
    for idx_img in range(cfg.batch_size):
        anchor = imgs_anchor[idx_img]
        pos = imgs_pos[idx_img]
        neg = imgs_neg[idx_img]
        print(anchor.shape)
        print(pos.shape)
        print(neg.shape)
        _ShowImg(anchor)
        _ShowImg(pos)
        _ShowImg(neg)
        break
    
    print('data size is {}'.format(reader.GetDataSize()))


def TestLFW():
    reader = LFWReader(dir_images=cfg.path_LFW)
    _TestTripletGenerator(reader)
    
def TestARFace():
    reader = ARFaceReader(dir_images=cfg.path_AR)
    _TestTripletGenerator(reader)
    
def TestPCD():
    reader = PCDReader(dir_images=cfg.path_PCD)
    _TestTripletGenerator(reader)
    
def TestPEAL():
    #reader1 = PEALReader(dir_images=path_Pose)
    #TestTripletGenerator(reader1)
    reader2 = PEALReader(dir_images=cfg.path_Accessory)
    _TestTripletGenerator(reader2)
    
def TestMix():
    reader_PCD = PCDReader(dir_images=cfg.path_PCD)
    reader_AR = ARFaceReader(dir_images=cfg.path_AR)
    reader_LFW = LFWReader(dir_images=cfg.path_LFW)
    reader_Pose = PEALReader(dir_images=cfg.path_Pose)
    reader_Accessory = PEALReader(dir_images=cfg.path_Accessory)
    reader = MixedReader([reader_PCD, reader_AR, reader_LFW, reader_Pose, reader_Accessory])
    _TestTripletGenerator(reader)
    
if __name__=='__main__':
    pass
    # TestLFW()
    # TestARFace()
    # TestPCD()
    # TestMix()
    TestPEAL()

    



