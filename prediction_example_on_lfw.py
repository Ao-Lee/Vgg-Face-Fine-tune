import numpy as np
from scipy.spatial.distance import cosine
from data import LFWSet, DataLoader
from sklearn.preprocessing import normalize
from sklearn import metrics
from tqdm import tqdm
from vggface import VggFace, preprocess_input
import cfg
'''
def vgg_face(weights_path=None):
    img = Input(shape=(3, 224, 224))

    pad1_1 = ZeroPadding2D(padding=(1, 1))(img)
    conv1_1 = Conv2D(64, (3, 3), activation='relu', name='conv1_1')(pad1_1)
    pad1_2 = ZeroPadding2D(padding=(1, 1))(conv1_1)
    conv1_2 = Conv2D(64, (3, 3), activation='relu', name='conv1_2')(pad1_2)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_2)

    pad2_1 = ZeroPadding2D((1, 1))(pool1)
    conv2_1 = Conv2D(128, (3, 3), activation='relu', name='conv2_1')(pad2_1)
    pad2_2 = ZeroPadding2D((1, 1))(conv2_1)
    conv2_2 = Conv2D(128, (3, 3), activation='relu', name='conv2_2')(pad2_2)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2_2)

    pad3_1 = ZeroPadding2D((1, 1))(pool2)
    conv3_1 = Conv2D(256, (3, 3), activation='relu', name='conv3_1')(pad3_1)
    pad3_2 = ZeroPadding2D((1, 1))(conv3_1)
    conv3_2 = Conv2D(256, (3, 3), activation='relu', name='conv3_2')(pad3_2)
    pad3_3 = ZeroPadding2D((1, 1))(conv3_2)
    conv3_3 = Conv2D(256, (3, 3), activation='relu', name='conv3_3')(pad3_3)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2))(conv3_3)

    pad4_1 = ZeroPadding2D((1, 1))(pool3)
    conv4_1 = Conv2D(512, (3, 3), activation='relu', name='conv4_1')(pad4_1)
    pad4_2 = ZeroPadding2D((1, 1))(conv4_1)
    conv4_2 = Conv2D(512, (3, 3), activation='relu', name='conv4_2')(pad4_2)
    pad4_3 = ZeroPadding2D((1, 1))(conv4_2)
    conv4_3 = Conv2D(512, (3, 3), activation='relu', name='conv4_3')(pad4_3)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_3)

    pad5_1 = ZeroPadding2D((1, 1))(pool4)
    conv5_1 = Conv2D(512, (3, 3), activation='relu', name='conv5_1')(pad5_1)
    pad5_2 = ZeroPadding2D((1, 1))(conv5_1)
    conv5_2 = Conv2D(512, (3, 3), activation='relu', name='conv5_2')(pad5_2)
    pad5_3 = ZeroPadding2D((1, 1))(conv5_2)
    conv5_3 = Conv2D(512, (3, 3), activation='relu', name='conv5_3')(pad5_3)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2))(conv5_3)

    flat = Flatten()(pool5)
    fc6 = Dense(4096, activation='relu', name='fc6')(flat)
    fc6_drop = Dropout(0.5)(fc6)
    fc7 = Dense(4096, activation=None, name='fc7')(fc6_drop)
    
    
    fc7_activation = Activation('relu')(fc7)
    fc7_drop = Dropout(0.5)(fc7_activation)
    out = Dense(2622, activation='softmax', name='fc8')(fc7_drop)

    model = Model(inputs=img, outputs=out)

    if weights_path:
        model.load_weights(weights_path)

    # Topless = Model(inputs=model.input, outputs=fc7)
    Topless = Model(inputs=model.input, outputs=fc7_activation)
    return Topless
'''

def GetLoader():
    lfw = LFWSet(cfg.dir_pairs, cfg.dir_images)
    dl = DataLoader(lfw, batch_size=8, shuffle=False)
    return dl
    
# input: array of size (Batch, dim) for both X and Y
# output: array of size(Batch)
def DistanceEuclidean(X, Y):
    diff = (normalize(X) - normalize(Y))
    return (diff**2).sum(axis=1)
    
def DistanceCosine(X, Y):
    size = X.shape[0]
    distance = [cosine(X[idx,:], Y[idx,:]) for idx in range(size)]
    return np.array(distance)
    
def AUC(distance, label):
    label_reverse = np.logical_not(label)
    return metrics.roc_auc_score(label_reverse, distance)
    
if __name__ == "__main__":
    dl = GetLoader()
    model = VggFace(weights='face', include_top=False)
    
    distances = []
    labels = []
    for data, label in tqdm(dl):
        img1 = preprocess_input(data['img1'])
        img2 = preprocess_input(data['img2'])
        embedding1 = model.predict(img1)
        embedding2 = model.predict(img2)
        distance = DistanceEuclidean(embedding1, embedding2)
        #distance = DistanceCosine(embedding1, embedding2)
        distances += distance.tolist()
        labels += label.tolist()
    distances = np.array(distances)
    labels = np.array(labels)
    
    auc = AUC(distances, labels)
    print(auc)
    
    
    