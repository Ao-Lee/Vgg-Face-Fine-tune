from os.path import join, isdir
from os import listdir
import matplotlib.pyplot as plt
import sys
import numpy as np

sys.path.append('..')
from vggface import VggFace
from utils import Verify, MergeImage, Inputs2ArrayImage
import cfg

threshold = 1.1

def _VerifyFolder(path_to_folder, model):
    paths = listdir(path_to_folder)
    assert len(paths)==2
    path01 = join(path_to_folder, paths[0])
    path02 = join(path_to_folder, paths[1])
    dist = Verify(path01, path02, model)
    result = 'same' if dist < threshold else 'different'
    color = (0,255,0) if dist < threshold else (255,0,0)
    
    img1 = Inputs2ArrayImage(path01, dtype=np.uint8)
    img2 = Inputs2ArrayImage(path02, dtype=np.uint8)
    img_merged = MergeImage(img1, img2, color=color)

    name1 = _GetPersonName(paths[0])
    name2 = _GetPersonName(paths[1])
    
    title = '(' + name1 + ')' + ' - ' + result +  ' - ' + '(' + name2 + ')'
    return img_merged, title

def _ShowImage(img, title):
    plt.figure()
    plt.imshow(img.astype('uint8'))
    plt.title(title)
    plt.show()
    plt.close()
        
def _GetPersonName(filename):
    # filename = 'Adolfo_Rodriguez_Saa_0001.jpg'
    return filename[:-9]

if __name__ == '__main__':
    root = 'imgs\\verification'
    output_file = 'example.jpg'
    model = VggFace(path=cfg.dir_model_v2, is_origin=True)
    folders = listdir(root)
    imgs = []
    titles = []
    
    folders = [folder for folder in folders if isdir(join(root, folder))]

    for folder in folders:
        img, title = _VerifyFolder(join(root, folder), model)
        imgs.append(img)
        titles.append(title)
    
    f,axes = plt.subplots(len(folders), 1, figsize=(10,16))
    for ax, img, title in zip(axes, imgs, titles):
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(title)
        
    #plt.show()
    #plt.savefig()
    #plt.close()
        
    
