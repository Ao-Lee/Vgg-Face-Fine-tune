import numpy as np
import os
import pandas as pd
import shutil
from tqdm import tqdm
from PIL import Image
from vggface import VggFace, preprocess_input

dir_data01 = 'E:\\DM\Face Business\\aligned\\true'
dir_data02 = 'E:\\DM\Face Business\\aligned\\false'
dir_prediction01 = 'E:\\DM\Face Business\\prediction\\true'
dir_prediction02 = 'E:\\DM\Face Business\\prediction\\false'
threshold = 0.409
model = VggFace()

def GetPCDDataFrame(dir_parant):
    def _GetAllFiles(path):
        directories = []
        for item in os.listdir(path):
            full_path = os.path.join(path, item)
            cond1 = os.path.isfile(full_path)
            cond2 = full_path.find('.jpeg') != (-1) or full_path.find('.jpg') != (-1)
            if cond1 and cond2:
                directories.append(full_path)
        if len(directories)!=2:
            return None
        else:
            return directories
            
    list_name = []
    list_path0 = []
    list_path1 = []
    for dir_sub in os.listdir(dir_parant):
        dir_full = os.path.join(dir_parant, dir_sub)
        if not os.path.isdir(dir_full):
            continue
        directories = _GetAllFiles(dir_full)
        if directories is None:
            continue
        list_name.append(dir_sub)
        list_path0.append(directories[0])
        list_path1.append(directories[1])
        
    #construct dataframe from list_name, list_path0, and list_path1
    df = pd.DataFrame([])
    df['name'] = pd.Series(list_name)
    df['path1'] = pd.Series(list_path0)
    df['path2'] = pd.Series(list_path1)
    return df

def Path2Image(path):
    im = Image.open(path)
    im = im.resize((224,224))
    im = np.array(im).astype(np.float32)
    return im

def GetResult(path):
    df = GetPCDDataFrame(path)

    prediction = []
    for idx in tqdm(range(len(df))):
        path1 = df.loc[idx,'path1']
        path2 = df.loc[idx,'path2']
        img1 = Path2Image(path1)
        img2 = Path2Image(path2)
        img1 = np.expand_dims(img1, axis=0)
        img2 = np.expand_dims(img2, axis=0)
        pair = np.concatenate([img1,img2], axis=0)
        pair = preprocess_input(pair)
        embeddings = model.predict_on_batch(pair)
        
        dist = ((embeddings[0] - embeddings[1])**2).sum()
        if dist > threshold:
            prediction.append('different person')
        else:
            prediction.append('same person')
    # construct result data frame
    result = pd.DataFrame([])
    result['name'] = df['name']
    result['result'] = pd.Series(prediction)
    return result

def CopyFiles(df, path_from, path_to):
    categories = ['different person', 'same person']
    for category in categories:
        print('creating file {}'.format(path_to + os.sep + category))
        mask = df['result'] == category
        df_selected = df[mask]
        df_selected.reset_index(drop=True,inplace=True)

        for idx in range(len(df_selected)):
            old_dir_full = path_from + os.sep + df_selected.loc[idx, 'name']
            new_dir_full = path_to + os.sep + category + os.sep + df.loc[idx, 'name']
            shutil.copytree(old_dir_full, new_dir_full)

def GenerateResultFolder(path_from, path_to):
    df = GetResult(path_from)
    CopyFiles(df, path_from, path_to)
    grouped = df['result'].groupby(df['result'])
    print(grouped.count())

if __name__ == '__main__':
    info = 'this file is deprecated, it will be removed from later version'
    print(info)
    GenerateResultFolder(dir_data01, dir_prediction01)
    # GenerateResultFolder(dir_data02, dir_prediction02)

    
