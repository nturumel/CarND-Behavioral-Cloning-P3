import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image_dataset_from_directory
import sklearn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras import regularizers, optimizers
from keras import utils as np_utils
from keras.callbacks import ModelCheckpoint

'''


model3.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=10

'''
TRAIN_FILE = R"./CarND-Behavioral-Cloning-P3/data/filenames_angles.csv"
IMG_DIR = 'CarND-Behavioral-Cloning-P3/data/IMG'
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
NB_EPOCH = 10

def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

def buildGenerator():
    train_label_df = pd.read_csv(TRAIN_FILE, delimiter = ',', header = None, names = ['id', 'score'])
    print (train_label_df)
    datagen = ImageDataGenerator(brightness_range = [0.2,1.0], channel_shift_range = 150.0,  validation_split = 0.25)
    print("getting train generator")
    train_generator = datagen.flow_from_dataframe(dataframe = train_label_df, directory = IMG_DIR, x_col = "id", y_col = "score", subset = "training", has_ext = True, class_mode = "raw", shuffle = True, target_size = IMG_SIZE, batch_size = BATCH_SIZE, save_to_dir = './augmented/train/', validate_filenames = False)
    print("getting validation generator")
    valid_generator = datagen.flow_from_dataframe(dataframe = train_label_df, directory = IMG_DIR, x_col = "id", y_col = "score", subset = "validation", has_ext = True, class_mode = "raw", shuffle = True, target_size = IMG_SIZE, batch_size = BATCH_SIZE, save_to_dir = './augmented/train/', validate_filenames = False)
    return train_generator, valid_generator

def buildModel():
    
    IMG_SHAPE = IMG_SIZE + (3,)
    
    # define key layers
    data_augmentation = tf.keras.Sequential([ tf.keras.layers.experimental.preprocessing.RandomContrast([0, 0.5])])
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
    base_model = tf.keras.applications.MobileNetV2(input_shape = IMG_SHAPE, include_top = False, weights = 'imagenet')
    converge = tf.keras.layers.GlobalAveragePooling2D()
    dropout = Dropout(0.2)
    activate = Activation('relu')

    # chaining the layers
    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = base_model(x, training = False)
    x = converge(x)
    x = dropout(x)
    x = Flatten()(x)
    x = Dense(100)(x)
    x = dropout(x)
    x = activate(x)
    x = Dense(10)(x)
    x = dropout(x)
    x = activate(x)
    outputs = Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    opt = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = opt, loss='mse')
    print(model.summary())
    return model

if __name__ == "__main__":
    train_generator, valid_generator = buildGenerator()
    # added debug code
    x,y = valid_generator.next()
    for i in range(0,1):
        image = x[i]
        plt.imshow(image.transpose(2,1,0))
        plt.show()

    model = buildModel()
    checkpoint = ModelCheckpoint("steering_prediction_model.h5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto', period=1)
    model.fit_generator(train_generator, steps_per_epoch = train_generator.samples // BATCH_SIZE, validation_data = valid_generator, validation_steps = valid_generator.samples // BATCH_SIZE, epochs = NB_EPOCH, callbacks = [checkpoint])



    '''
 
    flow_from_dataframe(
    dataframe, directory=None, x_col='filename', y_col='class', weight_col=None,
    target_size=(256, 256), color_mode='rgb', classes=None,
    class_mode='categorical', batch_size=32, shuffle=True, seed=None,
    save_to_dir=None, save_prefix='', save_format='png', subset=None,
    interpolation='nearest', validate_filenames=True, **kwargs
)
    '''