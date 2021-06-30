import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.optimize import curve_fit
import os 
import scipy  
from sympy import *
from scipy import stats
from scipy.stats import spearmanr
from scipy import stats
from sklearn.metrics import mean_squared_error
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model
from statistics import mean
import tensorflow.keras.backend as k
from tensorflow import keras
from keras.optimizers import Adam, SGD
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau ,Callback,TensorBoard
from keras.models import load_model
from tensorflow.keras import applications 
from keras.activations import softmax,sigmoid
import h5py
from PIL import Image
from scipy.stats import spearmanr,pearsonr
import sklearn
import tensorflow as tf
import time
from tensorflow.keras.layers import MaxPooling2D ,Dense,Concatenate ,Dropout ,Input,concatenate,Conv2D,Reshape,GlobalMaxPooling2D,Flatten,GlobalAveragePooling2D,AveragePooling2D,Lambda,MaxPooling2D,TimeDistributed, Bidirectional, LSTM

import argparse
import random
from tqdm import tqdm
import urllib.request






tf.keras.backend.clear_session()
start_time = time.time()

def download_features(dataset):
	if dataset == 1:
		URL1 = "http://openvvc.insa-rennes.fr/models/KonViD-1k/x_test_konvid.npy"
		URL2 = "http://openvvc.insa-rennes.fr/models/KonViD-1k/y_test_konvid.npy"
		filename1 = "./features/x_test_konvid.npy"
		filename2 = "./features/y_test_konvid.npy"
		urllib.request.urlretrieve (URL1 , filename1 )
		urllib.request.urlretrieve (URL2 , filename2 )

	if dataset == 2:
		URL1 = "http://openvvc.insa-rennes.fr/models/LIVE-VQC/x_test_live.npy"
		URL2 = "http://openvvc.insa-rennes.fr/models/LIVE-VQC/y_test_live.npy"
		filename1 = "./features/x_test_live.npy"
		filename2 = "./features/y_test_live.npy"
		urllib.request.urlretrieve (URL1 , filename1 )
		urllib.request.urlretrieve (URL2 , filename2 )











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


  model.add(Dense(256,activation='relu'))
  model.add(layers.Dropout(rate=0.5))
  
  model.add(layers.Dense(1))
  model.add(layers.Activation('linear'))
  model.compile(optimizer=optimizers.Adam(),loss='mse',metrics=['mae'])
  model.summary()
  return model

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
  # 4-parameter logistic function
  logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
  yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
  return yhat

  
def patch_dimension(x_train):
  a , b = x_train.shape
  return(int(b/2048))



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset',default='3',  type=int, help='dataset to test: 1: for konvid, 2: for live, 3: for custom')


  parser.add_argument('--input_final_model',  type=str, help='path to the model')
  parser.add_argument('--sp_model_weights',  type=str, help='path to the model')


  parser.add_argument(
        '--x_test',
        default='',
        type=str,
        help='path to test .npy file')
  parser.add_argument(
        '--n',
        default='30',
        type=int,
        help='number of frames per videos')
  args = parser.parse_args()

  if not os.path.exists('./features'):
    os.makedirs('./features')

  dataset = args.dataset

  download_features(dataset)

  if dataset ==1:
  	x_test = np.load('./features/x_test_konvid.npy')
  	y_test = np.load('./features/y_test_konvid.npy')
  	sp_model = 'res-bi-sp_koniq.h5'
  	f_model = 'konvid_2_bilstm_join.h5'
  	n = 30
  elif dataset ==2:
  	x_test = np.load('./features/x_test_live.npy')
  	y_test = np.load('./features/y_test_live.npy')
  	sp_model = 'res-bi-sp_koniq.h5'
  	f_model = 'live2_bilstm_join.h5'
  	n = 30
  elif dataset==3:
  	features_name = args.x_test.split('x_test')
  	x_test = np.load('./features/x_test'+features_name[-1])
  	y_test = np.load('./features/y_test'+features_name[-1])
  	n = args.n
  	sp_model = args.sp_model_weights
  	f_model = args.input_final_model





  

  patches = patch_dimension(x_test)


  In = Input((n,patches,2048))
 
  model = load_model('./models/' +sp_model )
  model.summary()
  for layer in model.layers:
    layer.trainable = True
  model_final = Model(inputs=model.input,outputs=model.layers[-3].output )

  model = build_model((n,patches,2048), model_final)
  model.load_weights('./models/' + f_model)



  nbr_test_video = int(x_test.shape[0]/n)
  l_test = y_test.shape[0]

  x_test = np.reshape(x_test,(nbr_test_video,n,patches,2048))
  yy = y_test.tolist()
  y_ss = []
  i =0
  while i <l_test :
    temp = yy[i:i+n]

    moy = mean(temp)
    y_ss.append(moy)
    i = i+n

  y_ss = np.array(y_ss)
  y_ss = np.reshape(y_ss, (nbr_test_video,1))


  y_p = model.predict(x_test)
  model.summary()
  y_p = np.reshape(y_p,(nbr_test_video)) * 5
  y_ss = np.reshape(y_ss,(nbr_test_video)) * 5
  beta_init = [np.max(y_ss), np.min(y_ss), np.mean(y_p), 0.5]
  popt, _ = curve_fit(logistic_func, y_p, y_ss, p0=beta_init, maxfev=int(1e8))
  y_pred_logistic = logistic_func(y_p, *popt)
  from scipy.stats import gaussian_kde
  xy = np.vstack([y_ss,y_p])
  z = gaussian_kde(xy)(xy)
  m = min(y_ss)
  l = len(y_ss)
  u = max(y_ss) +0.5
  x = np.linspace(m-0.2,u+0.2,num=l)
  ms = y_ss.tolist()
  kf = ms -logistic_func(ms, *popt)

  
  sig = np.std(kf)
  

  
  print('srocc = ',spearmanr(y_ss,y_p).correlation)
  print('plcc = ', stats.pearsonr(y_ss,y_pred_logistic)[0])
  try:
    KRCC = scipy.stats.kendalltau(y_ss, y_p)[0]
  except:
    KRCC = scipy.stats.kendalltau(y_ss, y_p, method='asymptotic')[0]
  print('krocc = ' , KRCC)
  
  
  print('rmse = ' , np.sqrt(mean_squared_error(y_ss,y_pred_logistic)))
  print("--- %s seconds ---" % (time.time() - start_time))


  plt.scatter(y_p,y_ss, s=10, marker='o', c=z)
  plt.plot(x, logistic_func(x, *popt), c='red',label=r'fitted $f(x)$',linewidth=1)
  plt.plot(x, logistic_func(x, *popt)+ 2*sig,'--' , c='red',label=r'$f(x) \pm  2  \sigma$',linewidth=1)
  plt.plot(x, logistic_func(x, *popt)- 2*sig,'--' , c='red',linewidth=1)
  plt.xlabel("Predicted Score")
  plt.ylabel("MOS")
  plt.legend()
  plt.grid()
  plt.savefig('./figures/mos_sroc =' + str(spearmanr(y_ss,y_p).correlation)+'.png')
  '''
  name_list = pd.read_csv('test.csv')
  name_list = name_list.values.tolist()
  gt = y_ss.tolist()
  pred = y_p.tolist()
  df = pd.DataFrame(list(zip(gt, pred)),
               columns =['MOS', 'Predicted values'])
  df.to_csv('konvid.csv', index=False) 
  '''
