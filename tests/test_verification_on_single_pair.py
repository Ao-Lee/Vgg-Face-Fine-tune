from os.path import join
from os import listdir


import sys
sys.path.append('..')
from vggface import VggFace
from utils import Verify
import cfg


def _CheckFolder(path_to_folder, model):
    paths = listdir(path_to_folder)
    assert len(paths)==2
    path01 = join(path_to_folder, paths[0])
    path02 = join(path_to_folder, paths[1])
    dist = Verify(path01, path02, model)
    return dist

if __name__ == '__main__':
    root = 'imgs\\verification'
    threshold = 1.1
    model = VggFace(path=cfg.dir_model_v2, is_origin=True)
    for folder in listdir(root):
        path_to_folder = join(root, folder)
        dist = _CheckFolder(path_to_folder, model)
        if dist>threshold:
            print('Different Person')
        else:
            print('Same Person')

    
