import numpy as np
import cv2
import os 
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from statistics import mean
import tensorflow.keras.backend as k
from sklearn.utils import shuffle
from tensorflow import keras
from keras.optimizers import Adam
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau ,Callback
from keras.models import load_model
import csv
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications 
import PIL
from keras.activations import softmax,sigmoid
import h5py
from PIL import Image
from keras.layers import Layer
from scipy.stats import spearmanr,pearsonr
import sklearn
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D ,Dense,Concatenate ,Dropout ,Input,concatenate,Conv2D,Reshape,GlobalMaxPooling2D,Flatten,GlobalAveragePooling2D,AveragePooling2D,Lambda,MaxPooling2D,TimeDistributed, Bidirectional, LSTM
from tensorflow.keras.models import Model 
from tensorflow.keras import layers
import argparse
from keras.regularizers import l2
import random
from tqdm import tqdm



tf.keras.backend.clear_session()
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # for using cpu 
#os.environ['CUDA_VISIBLE_DEVICES']=""


def build_model(patches):

  model = models.Sequential()

  model.add(Bidirectional(LSTM(64,return_sequences=True,
                               kernel_initializer='random_normal',recurrent_initializer='random_normal', 
                               dropout=0.4,recurrent_dropout=0),input_shape = ( patches,2048)))

  model.add(Bidirectional(LSTM(64,return_sequences=True, 
                               kernel_initializer='random_normal', 
                               recurrent_initializer='random_normal', dropout=0.4,recurrent_dropout=0)))
  
  model.add(Flatten())
  '''
  model.add(Dense(2048,activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)))
  model.add(layers.Dropout(rate=0.25))
  model.add(Dense(1024,activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)))
  model.add(layers.Dropout(rate=0.25))
  '''
  model.add(Dense(256,activation='relu', kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.001)))
  model.add(layers.Dropout(rate=0.5))
 
  model.add(layers.Dense(1))
  model.add(layers.Activation('linear'))
  model.compile(optimizer=optimizers.Adam(),loss='mse',metrics=['mae'])
  model.summary()
  return model

def patch_dimension(x_train):
  a , b = x_train.shape
  return(int(b/2048))




if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
        '--x_train',
        default='',
        type=str,
        help='path to train npy file')
        
  args = parser.parse_args()

  if not os.path.exists('./models'):
    os.makedirs('./models')

  features_name = args.x_train.split('x_train')


  x_train = np.load('./features/x_train'+features_name[-1])
  y_train = np.load('./features/y_train'+features_name[-1])

  x_test = np.load('./features/x_test'+features_name[-1])
  y_test = np.load('./features/y_test'+features_name[-1])



  md = ModelCheckpoint(filepath='./models/konvid_spatial_pooling.h5',
  monitor='val_loss', mode='min',
  save_best_only=True,verbose=1)
  rd = ReduceLROnPlateau(monitor='val_loss', 
                         factor=0.5, patience=30, min_lr=1e-7, verbose=2, mode='min')

  callb = [md,rd]
  patches = patch_dimension(x_train)
  model = build_model(patches)
  x_train = x_train.reshape(x_train.shape[0],patches,2048)
  x_test = x_test.reshape(x_test.shape[0],patches,2048)


  history = model.fit(x_train,y_train,batch_size=16,
                    epochs=200,validation_data=(x_test,y_test),verbose=2,callbacks=callb)








#python vgg16_kadid.py --image_dir /content/kadid10k/images --model_name firt.h5 --csv_file /content/kadid10k/dmos.csv --batch_size 4 --epochs 50 --num_patch 15 --init_FC 1 --weights /content/models/vgg16_init.h5 
