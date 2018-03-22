from os.path import join
image_size = 224
batch_size = 16
_dir_root = 'E:\\DM\\VGG-Face'
dir_images = join(_dir_root, 'aligned')
dir_pairs = join(_dir_root, 'pairsDevTest.txt')
dir_model = join(_dir_root, 'model', 'vgg-face-keras-fc-tensorflow.h5')
dir_model_tuned = join(_dir_root, 'model', 'vgg-face-keras-fc-tensorflow-tuned.h5')
dir_model_trim = join(_dir_root, 'model', 'vgg-face-keras-fc-tensorflow-tuned-trim.h5')