import os
from scipy import misc

#for each img in folder_source, align it and store the aligned img in folder_target
def _AlignFolder(folder_source, folder_target, align_func):
    unable = []
    for item in os.listdir(folder_source):
        input_path = os.path.join(folder_source, item)
        cond1 = os.path.isfile(input_path)
        cond2 = input_path.find('.jpeg') != (-1) or input_path.find('.jpg') != (-1)
        if not cond1 or not cond2:
            continue
        img = align_func(input_path)
        if img is None:
            unable.append(input_path)
            continue
        if not os.path.exists(folder_target):
            os.makedirs(folder_target)
        output_path = os.path.join(folder_target, item)
        misc.imsave(output_path, img)
    return unable
    
'''
return value is a list containing all imgs that are not able to align
source is a image folder, which should have structure similar to the following:
Aaron_Eckhart
    Aaron_Eckhart_0001.jpg

Aaron_Guiel
    Aaron_Guiel_0001.jpg

Aaron_Patterson
    Aaron_Patterson_0001.jpg

Aaron_Peirsol
    Aaron_Peirsol_0001.jpg
    Aaron_Peirsol_0002.jpg
    Aaron_Peirsol_0003.jpg
    Aaron_Peirsol_0004.jpg
'''
def AlignDatabase(source, target, align_func):
    unables = []
    for item in os.listdir(source):
        print('working on directory {}'.format(item))
        folder_source = os.path.join(source, item)
        folder_target = os.path.join(target, item)
        if not os.path.isdir(folder_source):
            continue
        unable = _AlignFolder(folder_source, folder_target, align_func)
        unables = unables + unable
    
    print('{} images are unable to align......'.format(len(unables)))
    for path in unables:
        print(path)
    return unables