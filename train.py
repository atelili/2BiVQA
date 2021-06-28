import numpy as np
import cv2
import os 
from keras import backend as K
from scipy.stats import spearmanr
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
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

from keras.activations import softmax,sigmoid
import h5py
from keras.layers import Layer
from scipy.stats import spearmanr,pearsonr
import sklearn
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D ,Dense,Concatenate ,Dropout ,Input,concatenate,Conv2D,Reshape,GlobalMaxPooling2D,Flatten,GlobalAveragePooling2D,AveragePooling2D,Lambda,MaxPooling2D,TimeDistributed, Bidirectional, LSTM

from tensorflow.keras import layers
import argparse
from tqdm import tqdm




tf.keras.backend.clear_session()
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
#os.environ['CUDA_VISIBLE_DEVICES']=""



                       
def build_model(batch_shape, model_final):



  model = models.Sequential()
  model.add(TimeDistributed(model_final,input_shape = batch_shape))
 
  model.add(Bidirectional(LSTM(64,return_sequences=True,kernel_initializer='random_normal',
    recurrent_initializer='random_normal', 
                               dropout=0.4,recurrent_dropout=0)))
  model.add(Bidirectional(LSTM(64,return_sequences=True, 
                               kernel_initializer='random_normal', 
                               recurrent_initializer='random_normal', dropout=0.4,recurrent_dropout=0)))
  
  model.add(Flatten())

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

  parser.add_argument(
        '--n',
        default=32,
        type=int,
        help='number of frames per video') 

  parser.add_argument(
        '--spatial_weights',
        default='',
        type=str,
        help='sp model weights') 
        
  args = parser.parse_args()

  if not os.path.exists('./models'):
    os.makedirs('./models')

  features_name = args.x_train.split('x_train')


  x_train = np.load('./features/x_train'+features_name[-1])
  y_train = np.load('./features/y_train'+features_name[-1])

  x_test = np.load('./features/x_test'+features_name[-1])
  y_test = np.load('./features/y_test'+features_name[-1])

  md = ModelCheckpoint(filepath='./models/konvid_2_bilstm.h5',
  monitor='val_loss', mode='min',save_weights_only=True,
  save_best_only=True,verbose=1)

  rd = ReduceLROnPlateau(monitor='val_loss', 
                         factor=0.5, patience=20,min_lr=1e-7, verbose=2, mode='min')

  callb = [md,rd]
  

  
  n = args.n
  nbr_train_video = int(x_train.shape[0]/n)
  nbr_test_video = int(x_test.shape[0]/n)

  patches = patch_dimension(x_train)
  l_train = y_train.shape[0]
  l_test = y_test.shape[0]

  x_train = np.reshape(x_train, (nbr_train_video,n,patches,2048))
  x_test = np.reshape(x_test,(nbr_test_video,n,patches,2048))
  yy = y_test.tolist()
  yr = y_train.tolist()
  y_ss = []
  y_tr = []
  i =0
  while i <l_test :
    temp = yy[i:i+n]
    moy = mean(temp)
    y_ss.append(moy)
    i = i+n
  i = 0
  
  while i <l_train :
    temp = yr[i:i+n]
    moy = mean(temp)
    y_tr.append(moy)
    i = i+n
    
  y_ss = np.array(y_ss)
  y_ss = np.reshape(y_ss, (nbr_test_video,1))
  
  y_tr = np.array(y_tr)
  y_tr = np.reshape(y_tr, (nbr_train_video,1))

  In = Input((n,patches,2048))
  model = load_model('./models/' + args.spatial_weights)
  for layer in model.layers:
    layer.trainable = True
  model_final = Model(inputs=model.input,outputs=model.layers[-3].output )
  model = build_model((n,patches,2048), model_final)
  
  history = model.fit(x_train,y_tr,batch_size=16,
                    epochs=200,validation_data=(x_test,y_ss),verbose=2,callbacks=callb)




