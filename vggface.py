from keras.models import Model
from keras.layers import Lambda, Activation, Input, Conv2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
from keras import backend as K
import cfg
assert cfg.image_size==224

def VggFace(path=cfg.dir_model, include_top=False):
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

    if path is not None:
        model.load_weights(path)

    if include_top==True:
        return model
    else:
        norm = Lambda(lambda x: K.l2_normalize(x, axis=1))(fc7)
        topless = Model(inputs=model.input, outputs=norm)
        return topless
        
def preprocess_input(batch_image):
    if not isinstance(batch_image, (np.ndarray, np.generic)):
        error_msg = "data must be 4d numpy array, but found {}"
        raise TypeError(error_msg.format(type(batch_image)))
    shape = batch_image.shape
    if len(shape) != 4:
        error_msg = "data must be shape of (batch, 224, 224, 3), but found {}"
        raise ValueError(error_msg.format(shape))
    (batch, size0, size1, channel) = shape
    if size0 != 224 or size1 != 224 or channel != 3:
        error_msg = "data must be shape of (batch, 224, 224, 3), but found {}"
        raise ValueError(error_msg.format(shape))
        
    batch_image = batch_image.astype(np.float32)
    batch_image = batch_image.transpose([0,3,1,2])
    return batch_image
    