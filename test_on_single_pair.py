import numpy as np
from os.path import isfile
from sklearn.preprocessing import normalize
from vggface import VggFace, preprocess_input
from PIL import Image

threshold = 0.4
path01 = 'E:\\DM\VGG-Face\\aligned\Adel_Al-Jubeir\\Adel_Al-Jubeir_0001.jpg'
path02 = 'E:\\DM\VGG-Face\\aligned\Adel_Al-Jubeir\\Adel_Al-Jubeir_0002.jpg'
model = VggFace()


def Path2Image(path):
    im = Image.open(path)
    im = im.resize((224,224))
    im = np.array(im).astype(np.float32)
    return im
    
def DistanceEuclidean(X, Y):
    X = X.reshape(1, -1)
    Y = Y.reshape(1, -1)
    diff = (normalize(X) - normalize(Y))
    return (diff**2).sum()
    

if __name__ == '__main__':
    assert(isfile(path01))
    assert(isfile(path02))
    img1 = Path2Image(path01)
    img2 = Path2Image(path02)
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)
    pair = np.concatenate([img1,img2], axis=0)
    pair = preprocess_input(pair)
    embeddings = model.predict_on_batch(pair)
    dist = DistanceEuclidean(embeddings[0], embeddings[1])
    print(dist)
    if dist>threshold:
        print('D')
    else:
        print('S')
    

    
