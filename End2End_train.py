"""
Author : 
    Ahmed Telili
"""

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
from keras.optimizers import Adam, SGD
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





tf.keras.backend.clear_session()
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
#os.environ['CUDA_VISIBLE_DEVICES']=""



def data_generator(data,batch_size=16):              

    num_samples = len(data)
    random.shuffle(data)

    while True:   
        for offset in range(0, num_samples, batch_size):
        	
            # Get the samples you'll use in this batch
            batch_samples = data[offset:offset+batch_size]
            X_train = np.zeros((batch_size, 30,25,2560))
            y_train = np.zeros((batch_size,1))
            for i in range(batch_size):
              X_train[i,:,:,:] = np.load(batch_samples[i][0])
              y_train[i,:] = np.load(batch_samples[i][1])
              y_train[i,:] =  y_train[i,:]


              
            yield X_train, y_train



def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
  # 4-parameter logistic function
  logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
  yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
  return yhat

'''
def data_generator_1(data,batch_size=4):              

    num_samples = len(data)

    while True:   
        for offset in range(0, num_samples, batch_size):
          
            # Get the samples you'll use in this batch
            batch_samples = data[offset:offset+batch_size]
            X_train = np.zeros((batch_size, 30,25,2560))
            y_train = np.zeros((batch_size,1))
            for i in range(batch_size):
              X_train[i,:,:,:] = np.load(batch_samples[i][0])
              y_train[i,:] = np.load(batch_samples[i][1])
            yield X_train


def data_generator_2(data,batch_size=1):              

    num_samples = len(data)

    while True:   
        for offset in range(0, num_samples, batch_size):
          
            # Get the samples you'll use in this batch
            batch_samples = data[offset:offset+batch_size]
            X_train = np.zeros((batch_size, 30,25,2560))
            y_train = np.zeros((batch_size,1))
            for i in range(batch_size):
              X_train[i,:,:,:] = np.load(batch_samples[i][0])
              y_train[i,:] = np.load(batch_samples[i][1])
            yield y_train
'''


                      
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
	x = os.listdir('features_X')
	li = []
	for i in range(len(x)):
		tem = []
		x_f = './features_X/' + x[i]
		y_f = './features_y/' + x[i]
		tem.append(x_f)
		tem.append(y_f)
		li.append(tem)
	li.sort()

	return (li)




if __name__ == '__main__':
	parser = argparse.ArgumentParser("End2End_train")

	parser.add_argument('-np',
        '--num_patch',
        default=15,
        type=int,
        help='Number of cropped patches per frames.'
    )
	parser.add_argument('-nf',
        '--num_frames',
        default=30,
        type=int,
        help='Number of cropped frames per video.'
    )
	parser.add_argument('-m',
        '--pretrained_model',
        default='/models/res-bi-sp_koniq.h5',
        type=str,
        help='path to pretrained spatial pooling module.'
    )




  if not os.path.exists('./models'):
    os.makedirs('./models')

  #train_list, val_list = prepare_datalist(path_to_csv = args.csv_file , images_dir= args.image_dir, random_split=False)

  #num_patch = args.num_patch
  #batch_size = args.batch_size
  #batch_shapes = (num_patch,224,224,3)
  
  md = ModelCheckpoint(filepath='./models/trained_model.h5',monitor='val_loss', mode='min',save_weights_only=True,save_best_only=True,verbose=1)
  rd = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20,min_lr=1e-7, verbose=2, mode='min')
  ear = EarlyStopping(monitor='val_loss',mode ='min', patience=80, verbose=2,restore_best_weights=False)
  callbacks_k = [md,rd,TqdmCallback(verbose=2),ear]
  li = data_prepare()
  li.sort()
  num_patch = args.num_patch
  nb = args.num_frames
  sp_pretrained = args.pretrained_model
  train_l = li[0:960]
  test_l = li[960:1200]
  train_gen = data_generator(train_l,batch_size=16)
  val_gen = data_generator(test_l,batch_size=16)
  In = Input((nb,num_patch,2048))
  model = load_model(sp_pretrained)
  for layer in model.layers:
  	layer.trainable = True
  model_final = Model(inputs=model.input,outputs=model.layers[-3].output )
  model = build_model((nb,num_patch,2048), model_final)

  history = model.fit_generator(train_gen,steps_per_epoch = int(len(train_l)/batch_size),
		epochs=200,validation_data=val_gen,validation_steps =
		int(len(test_l)/batch_size) ,verbose=0,callbacks=callbacks_k)

  	
