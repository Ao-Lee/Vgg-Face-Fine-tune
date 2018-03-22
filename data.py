'''
Note: All images should be aligend first
'''
import pandas as pd
import numpy as np
import os
from os import listdir
from os.path import join, exists
from PIL import Image
from vggface import preprocess_input
import matplotlib.pyplot as plt

try:
    from UtilsData import Dataset, DataLoader
except ImportError:
    from .UtilsData import Dataset, DataLoader
import cfg

'''
self.df has following structure:
name1           dir1            name2       dir2
Aaron_Eckhart   F:\\Pic01.jpg   Abba_Eban   F:\\Pic02.jpg
Abba_Eban       F:\\Pic02.jpg   Abba_Eban   F:\\Pic03.jpg
Abdullah        F:\\Pic04.jpg   Abba_Eban   F:\\Pic05.jpg
'''
class LFWSet(Dataset):
    def __init__(self, dir_pairs, dir_images):
        pairs = self._ReadPairs(dir_pairs)
        self.df = self._GenerateDataFrame(dir_images, pairs, suffix='jpg')
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        data = {}
        path1 = self.df.loc[idx,'dir1']
        path2 = self.df.loc[idx,'dir2']  
        data['img1'] = self.Trans(path1)
        data['img2'] = self.Trans(path2)
        data['name1'] = self.df.loc[idx,'name1']
        data['name2'] = self.df.loc[idx,'name2']
        label = 1 if data['name1']==data['name2'] else 0
        return data, label
     
    @staticmethod
    # convert path to image
    def Trans(path):
        im = Image.open(path)
        im = im.resize((224,224))
        im = np.array(im).astype(np.float32)
        return im
    
    @staticmethod
    def _GenerateDataFrame(dir_images, pairs, suffix='jpg'):
        def Name2Dir(name, which):
            return os.path.join(dir_images, name, name + '_' + '%04d' % int(which)+'.'+suffix)
        def ConvertPair(pair):
            name1 = pair[0]
            dir1 = Name2Dir(name1,pair[1])
            name2 = pair[2]
            dir2 = Name2Dir(name2,pair[3])
            return [name1, dir1, name2, dir2]
            
        pairs = [ ConvertPair(pair) for pair in pairs]
        pairs_available = [pair for pair in pairs if os.path.exists(pair[1]) and os.path.exists(pair[3])]
        
        skipped_pairs = len(pairs) - len(pairs_available)
        
        if skipped_pairs>0:
            print('Skipped %d image pairs' % skipped_pairs)
    
        df = pd.DataFrame(pairs_available, columns=['name1', 'dir1', 'name2', 'dir2'])
        return df
    
    @staticmethod
    def _ReadPairs(dir_pairs):
        pairs = []
        with open(dir_pairs, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                if len(pair)==3:
                    pair.insert(2, pair[0])
                pairs.append(pair)
        return np.array(pairs)

def TestLFWSet():
    lfw = LFWSet(cfg.dir_pairs, cfg.dir_images)
    dl = DataLoader(lfw, batch_size=4, shuffle=True)
    
    for data, label in dl:
        break
    print(data['img1'].shape)
    print(data['img2'].shape)
    print(label.shape)
    
class PCDReader(object):
    def __init__(self, dir_images):
        self.root = dir_images
        self.root = 'E:\\DM\\PCD\\aligned'
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
        
class ARFaceReader(object):
    def __init__(self, dir_images):
        self.root = dir_images
 
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
        
class LFWReader(object):
    def __init__(self, dir_images):
        self.root = dir_images
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
    reader_PCD = PCDReader(dir_images='E:\\DM\\PCD\\aligned')
    reader_AR = ARFaceReader(dir_images='E:\\DM\\ARFace\\aligned')
    reader_LFW = LFWReader(dir_images='E:\\DM\\VGG-Face\\aligned')
    reader = MixedReader([reader_PCD, reader_AR, reader_LFW])
'''
class MixedReader(object):
    def __init__(self, list_readers):
        self.readers = list_readers
        
    def GetTriplet(self):
        idx = np.random.randint(low=0, high=len(self.readers))
        path_anchor, path_pos, path_neg = self.readers[idx].GetTriplet()
        return path_anchor, path_pos, path_neg
        
def ReadAndResize(filepath):
    im = Image.open((filepath)).convert('RGB')
    im = im.resize((cfg.image_size, cfg.image_size))
    return np.array(im, dtype="float32")

def Flip(im_array):
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
            img_anchor = Flip(ReadAndResize(path_anchor))
            img_pos = Flip(ReadAndResize(path_pos))
            img_neg = Flip(ReadAndResize(path_neg))
            list_pos.append(img_pos)
            list_anchor.append(img_anchor)
            list_neg.append(img_neg)

        A = preprocess_input(np.array(list_anchor))
        P = preprocess_input(np.array(list_pos))
        N = preprocess_input(np.array(list_neg))
        label = None
        
        yield ({'anchor_input': A, 'positive_input': P, 'negative_input': N}, label)        

def ShowImg(img):
    plt.figure()
    plt.imshow(img.astype('uint8'))
    plt.show()
    plt.close()
    
def TestTripletGenerator(reader):  
    # reader = LFWReader(dir_images='E:\\DM\\VGG-Face\\aligned')
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

        ShowImg(anchor)
        ShowImg(pos)
        ShowImg(neg)
        break

def TestLFW():
    reader = LFWReader(dir_images='E:\\DM\\VGG-Face\\aligned')
    TestTripletGenerator(reader)
    
def TestARFace():
    reader = ARFaceReader(dir_images='E:\\DM\\ARFace\\aligned')
    TestTripletGenerator(reader)
    
def TestPCD():
    reader = PCDReader(dir_images='E:\\DM\\PCD\\aligned')
    TestTripletGenerator(reader)
    
def TestMix():
    reader_PCD = PCDReader(dir_images='E:\\DM\\PCD\\aligned')
    reader_AR = ARFaceReader(dir_images='E:\\DM\\ARFace\\aligned')
    reader_LFW = LFWReader(dir_images='E:\\DM\\VGG-Face\\aligned')
    reader = MixedReader([reader_PCD, reader_AR, reader_LFW])
    print(reader)
    TestTripletGenerator(reader)
    
if __name__=='__main__':
    # TestLFW()
    # TestARFace()
    # TestPCD()
    TestMix()

    



