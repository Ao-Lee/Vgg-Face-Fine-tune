import numpy as np
from scipy.spatial.distance import cosine
from sklearn.preprocessing import normalize
from sklearn import metrics
from tqdm import tqdm

from vggface import VggFace, preprocess_input
from data import LFWSet, DataLoader
import cfg

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
    
def GetBestThreshold(distance, label):
    def Cost(fpr, tpr):
        return (1-tpr) + fpr

    label_reverse = np.logical_not(label)
    fpr, tpr, thresholds = metrics.roc_curve(label_reverse, distance)
    
    cost = np.inf
    threshold = None
    for idx in range(len(thresholds)):
        current_cost = Cost(fpr[idx], tpr[idx])
        if current_cost < cost:
            cost = current_cost
            threshold = thresholds[idx]
    return threshold, cost
        

if __name__ == "__main__":
    info = 'this file is deprecated, it will be removed from later version'
    print(info)
    dl = GetLoader()
    model = VggFace()
    # model = VggFace(path=None)
    # model.load_weights(cfg.dir_model_tuned)
    
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
    threshold, _ = GetBestThreshold(distances, labels)
    prediction = (distances < threshold).astype(np.int)
    acc = metrics.accuracy_score(labels, prediction)
    print('auc is: {}'.format(auc))
    print('threshold is {}'.format(threshold)) # 0.409
    print('accuracy is {}'.format(acc))
    
    
    