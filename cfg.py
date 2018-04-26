from os.path import join
image_size = 224
batch_size = 16


_dir_models = 'E:\\DM\\Faces_origin\\Models'
dir_model = join(_dir_models, 'vgg-face-keras-fc-tensorflow.h5')
dir_model_v2 = join(_dir_models, 'vgg-face-keras-fc-tensorflow-v2.h5')



path_LFW = 'E:\\DM\\Faces\\Data\\LFW\\aligned'
path_AR = 'E:\\DM\\Faces\\Data\\ARFace\\aligned'
path_PCD = 'E:\\DM\\Faces\\Data\\PCD\\aligned'
path_Pose = 'E:\\DM\\Faces\\Data\\PEAL_Pose'
path_Accessory = 'E:\\DM\\Faces\\Data\\PEAL_Accessory'