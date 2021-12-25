import numpy as np
import matplotlib.pyplot as plt
import cv2
import os 
from keras import backend as K
from tqdm.keras import TqdmCallback
from scipy.stats import spearmanr
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from statistics import mean
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import pandas as pd
import datetime
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau ,Callback,TensorBoard
from keras.models import load_model
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
import argparse
import random
from tqdm import tqdm
import time
from scipy.optimize import curve_fit




tf.keras.backend.clear_session()
start_time = time.time()

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
  # 4-parameter logistic function
  logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
  yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
  return yhat


def data_generator_1(data, nb, batch_size=1):              

    num_samples = len(data)

    while True:   
        for offset in range(0, num_samples, batch_size ):
          
            # Get the samples you'll use in this batch
            batch_samples = data[offset:offset+batch_size]
            X_train = np.zeros((batch_size, nb,25,2048))
            y_train = np.zeros((batch_size,1))
            for i in range(batch_size):
              X_train[i,:,:,:] = np.load(batch_samples[i][0])
              y_train[i,:] = np.load(batch_samples[i][1])
            yield X_train


def data_generator_2(data, nb ,batch_size=1):              

    num_samples = len(data)

    while True:   
        for offset in range(0, num_samples, batch_size):
          
            # Get the samples you'll use in this batch
            batch_samples = data[offset:offset+batch_size]
            X_train = np.zeros((batch_size, nb,25,2048))
            y_train = np.zeros((batch_size,1))
            for i in range(batch_size):
              X_train[i,:,:,:] = np.load(batch_samples[i][0])
              y_train[i,:] = np.load(batch_samples[i][1])
            yield y_train

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

def data_prepare():
  x = os.listdir('./features_X/')
  li = []
  for i in range(len(x)):
    tem = []
    x_f =  './features_X/' + x[i]
    y_f =  './features_y/' + x[i]
    tem.append(x_f)
    tem.append(y_f)
    li.append(tem)
  li.sort()

  return (li)

if __name__ == '__main__':
  parser = argparse.ArgumentParser("Demo")


  parser.add_argument('-nf',
        '--num_frames',
        default=30,
        type=int,
        help='Number of cropped frames per video.'
    )

  parser.add_argument('-m',
        '--pretrained_model',
        default='',
        type=str,
        help='path to pretrained End2End module.'
    ) 
  parser.add_argument('-f',
        '--paths',
        default='',
        type=str,
        help='path to videos features.'
    ) 

  args = parser.parse_args()

  model_sp = './models/res-bi-sp_koniq.h5'
  nb = args.num_frames
  model_end = args.pretrained_model
  paths = args.paths
  model = load_model(model_sp)
  model_final = Model(inputs=model.input,outputs=model.layers[-3].output )
  model = build_model((nb,25,2048), model_final)
  model.load_weights(model_end)
  test_l = data_prepare()

  test_gen = data_generator_1(test_l,nb)
  
  s_gen = data_generator_2(test_l,nb)
  y_p = model.predict(test_gen, steps = int(len(test_l)))
  y_ss = []

  for i in range(len(test_l)):
     y = next(s_gen)
     y_ss.append(y)
  y_ss = np.array(y_ss)
  y_ss = y_ss.reshape(len(test_l),1)
  srocc = spearmanr(y_ss,y_p).correlation
  y_p = np.reshape(y_p,(len(test_l))) 
  y_p =  y_p * 5
  y_ss = np.reshape(y_ss,(len(test_l))) 
  names = []
  for i in range(len(test_l)):
    a = test_l[i][0].split('/')[-1]
    a = a.split('.npy')[0]
    names.append(a)
  
  y_ss_l = y_ss.tolist()
  y_p_l = y_p.tolist()
  df = pd.DataFrame(list(zip(names, y_ss_l,y_p_l)),
               columns =['Name', 'MOS', 'Predicted MOS'])

  df.to_csv('results.csv', index=False)
  
  beta_init = [np.max(y_ss), np.min(y_ss), np.mean(y_p), 0.5]
  popt, _ = curve_fit(logistic_func, y_p, y_ss, p0=beta_init, maxfev=int(1e8))
  y_pred_logistic = logistic_func(y_p, *popt)
      
  plcc = stats.pearsonr(y_ss,y_pred_logistic)[0]
  rmse = np.sqrt(mean_squared_error(y_ss,y_pred_logistic))
  try:
       KRCC = scipy.stats.kendalltau(y_ss, y_p)[0]
  except:
       KRCC = scipy.stats.kendalltau(y_ss, y_p, method='asymptotic')[0]
  rmse = np.sqrt(mean_squared_error(y_ss,y_pred_logistic))
  print('srocc = ', srocc )
  print('plcc = ' , plcc)
  print( 'rmse = ', rmse)
  print('krocc = ', KRCC)
 
