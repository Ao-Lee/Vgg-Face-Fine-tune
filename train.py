import numpy as np
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input
# from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.preprocessing import normalize
from vggface import VggFace
import cfg
from data import LFWReader, ARFaceReader, PCDReader, MixedReader, PEALReader
from data import TripletGenerator

def triplet_loss(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = K.square(anchor - positive)
    negative_distance = K.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = K.sqrt(K.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = K.sqrt(K.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = K.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = K.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = K.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = K.log(1 + K.exp(loss))
    return K.mean(loss)

    
def triplet_loss_np(inputs, dist='sqeuclidean', margin='maxplus'):
    anchor, positive, negative = inputs
    positive_distance = np.square(anchor - positive)
    negative_distance = np.square(anchor - negative)
    if dist == 'euclidean':
        positive_distance = np.sqrt(np.sum(positive_distance, axis=-1, keepdims=True))
        negative_distance = np.sqrt(np.sum(negative_distance, axis=-1, keepdims=True))
    elif dist == 'sqeuclidean':
        positive_distance = np.sum(positive_distance, axis=-1, keepdims=True)
        negative_distance = np.sum(negative_distance, axis=-1, keepdims=True)
    loss = positive_distance - negative_distance
    if margin == 'maxplus':
        loss = np.maximum(0.0, 1 + loss)
    elif margin == 'softplus':
        loss = np.log(1 + np.exp(loss))
    return np.mean(loss)

def check_loss():
    batch_size = 10
    shape = (batch_size, 4096)

    p1 = normalize(np.random.random(shape))
    n = normalize(np.random.random(shape))
    p2 = normalize(np.random.random(shape))
    
    input_tensor = [K.variable(p1), K.variable(n), K.variable(p2)]
    out1 = K.eval(triplet_loss(input_tensor))
    input_np = [p1, n, p2]
    out2 = triplet_loss_np(input_np)

    assert out1.shape == out2.shape
    print(np.linalg.norm(out1))
    print(np.linalg.norm(out2))
    print(np.linalg.norm(out1-out2))

    
def GetModel():
    embedding_model = VggFace(is_origin=True)
    input_shape = (3, cfg.image_size, cfg.image_size)
    anchor_input = Input(input_shape, name='anchor_input')
    positive_input = Input(input_shape, name='positive_input')
    negative_input = Input(input_shape, name='negative_input')
    anchor_embedding = embedding_model(anchor_input)
    positive_embedding = embedding_model(positive_input)
    negative_embedding = embedding_model(negative_input)

    inputs = [anchor_input, positive_input, negative_input]
    outputs = [anchor_embedding, positive_embedding, negative_embedding]
       
    triplet_model = Model(inputs, outputs)
    triplet_model.add_loss(K.mean(triplet_loss(outputs)))
    return embedding_model, triplet_model
    
if __name__=='__main__':
    
    reader_PCD = PCDReader(dir_images=cfg.path_PCD)
    reader_AR = ARFaceReader(dir_images=cfg.path_AR)
    reader_LFW = LFWReader(dir_images=cfg.path_LFW)
    reader_Pose = PEALReader(dir_images=cfg.path_Pose)
    reader_Accessory = PEALReader(dir_images=cfg.path_Accessory)
    reader = MixedReader([reader_PCD, reader_AR, reader_LFW, reader_Pose, reader_Accessory])
    
    reader_tr = MixedReader([reader_LFW, reader_Pose, reader_Accessory])
    reader_te = MixedReader([reader_PCD, reader_AR])

    gen_tr = TripletGenerator(reader_tr)
    gen_te = TripletGenerator(reader_te)
    embedding_model, triplet_model = GetModel()
    
    
    
    
    for layer in embedding_model.layers[-2:]:
        layer.trainable = True
    for layer in embedding_model.layers[:-2]:
        layer.trainable = False
        
    triplet_model.compile(loss=None, optimizer=Adam(0.00002))
    history = triplet_model.fit_generator(gen_tr, 
                              validation_data=gen_te,  
                              epochs=5, 
                              verbose=1, 
                              workers=4,
                              steps_per_epoch=100, 
                              validation_steps=50)
    
    
    
    for layer in embedding_model.layers[30:]:
        layer.trainable = True
    for layer in embedding_model.layers[:30]:
        layer.trainable = False
        
    triplet_model.compile(loss=None, optimizer=Adam(0.000003))
    
    history = triplet_model.fit_generator(gen_tr, 
                              validation_data=gen_te,  
                              epochs=1, 
                              verbose=1, 
                              workers=4,
                              steps_per_epoch=500, 
                              validation_steps=20)
    
    embedding_model.save_weights(cfg.dir_model_tuned)
    '''
    
    