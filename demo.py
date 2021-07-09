import numpy as np
import cv2
import os
import time
import shutil
from tensorflow.keras import layers
from keras.models import load_model
from keras.layers import Layer
from tensorflow.keras.layers import MaxPooling2D ,Dense,Concatenate ,Dropout ,Input,concatenate,Conv2D,Reshape,GlobalMaxPooling2D,Flatten,GlobalAveragePooling2D,AveragePooling2D,Lambda,MaxPooling2D,TimeDistributed, Bidirectional, LSTM
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from sklearn.utils import shuffle
from tensorflow import keras
from keras.optimizers import Adam
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications
import h5py
from PIL import Image
from keras.layers import Layer
import tensorflow as tf
import argparse
from tqdm import tqdm
from utils.utils import *



tf.keras.backend.clear_session()


def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
            counter += 1
    return points

def random_crop(img, shape):
    return tf.image.random_crop(img, shape)

def crop_image_2(img, overlapping,num_patch):

    img_h, img_w, _ = img.shape
    split_width = 224
    split_height = 224
    X_points = start_points(img_w, split_width, overlapping)
    Y_points = start_points(img_h, split_height,overlapping )

    count = 0
    imgs = []


    for i in Y_points:
        for j in X_points:
            split = img[i:i+split_height, j:j+split_width]
            imgs.append(split)
            count += 1



    if len(X_points)*len(Y_points) < num_patch:
        dif = num_patch - len(X_points)*len(Y_points)
        for i in range(dif) :
            imgs.append(random_crop(img,(224,224,3)).numpy())

    elif len(X_points)*len(Y_points) > num_patch:
        imgs = imgs[0:num_patch]

    




    return(imgs)


def convert(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
      
    return "%d:%02d:%02d" % (hour, minutes, seconds)
      



class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size=1, patches = 25,
                  shuffle=False, list_IDs='',overlapping = 0.2, nb = 30):
        'Initialization'
        self.batch_size = batch_size
        self.nb = nb
        self.patches = patches
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.overlapping = overlapping
        self.on_epoch_end()
   
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs)/ self.batch_size))

    def __getitem__(self, index, nb = 30 ):
        
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        batch = [self.list_IDs[k] for k in indexes]
        
        # Generate data
        name, X = self.__data_generation(batch, nb)

        return name, X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, batch, nb=30):

        # Initialization
        X = np.empty((self.nb, self.patches,224,224, 3))
        
        
        # Generate data
        for i, ID in enumerate(batch):

            imgs = TemporalCrop(ID, self.nb)

            for k in range(len(imgs)):
                im = crop_image_2(imgs[k],overlapping= self.overlapping, num_patch= self.patches)
                for j in range(self.patches):
                    im = np.array(im)
                    X[k,j,:,:,:]=im[j,:,:,:]
                    X[k,j,:,:,:] = tf.keras.applications.resnet50.preprocess_input(X[k,j,:,:,:])
            name = ID
        
                   
              
        return name, X




def TemporalCrop(input_video_path, nb):
    out = []
    final = []
    
    cap = cv2.VideoCapture(input_video_path)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    
    while(cap.isOpened()):
        
            ret, frame = cap.read()
            if ret:

                out.append(frame)
            else:
                break
    step = int(N/nb)

    i = 0
    j = 0
    while i < nb :

        img = out[j]
        final.append(img)
        j = j +step
        i = i +1
        
        
        
    return(final)





def extract_features(model,list_IDs, samples, batch_size=1, num_patch = 25,overlapping= 0.2):
    videos = DataGenerator(batch_size=batch_size, list_IDs=list_IDs, patches = num_patch, overlapping = overlapping)
    name = []
    features_X = np.zeros((samples,num_patch,2048))
    i=0 
    for ID,X in tqdm(videos):
        for l in range(samples):
            features = model.predict(X[l,:,:,:,:])
            features_X[l,:,:] = features
            

        ID = ID.split('.')[0]
        ID = ID.split('/')[-1]
        np.save('./features/'+ID,features_X)

        




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
  return model


def patch_dimension(x_train):
  a , b = x_train.shape
  return(int(b/2048))




if __name__ == '__main__':

    parser = argparse.ArgumentParser("features_extracion")

    parser.add_argument(
        '--video_dir',
        default='',
        type=str,
        help='Directory path of frames')







    args = parser.parse_args()
    if not os.path.exists('./features'):
        os.makedirs('./features')

    video_dir = args.video_dir




    test_list = os.listdir(video_dir)

    for i in range(len(test_list)):
        test_list[i] = video_dir + '/' + test_list[i]



    num_patch = 25
    overlap = 0.2
    n = 30

    batch_shapes = (num_patch,224,224,3)

    model_final = model_build(batch_shapes)
    print('======================================================')
    start_time = time.time()
    print('Extract features ...')



    extract_features(model_final,test_list,samples =n, batch_size=1,  num_patch = num_patch,overlapping= overlap)
    print('======================================================')
    print('Done! ')
    t1 = time.time() - start_time
    print('Time needed to extract features: ',convert(t1))

    print('======================================================')
    t2 = time.time()
    In = Input((n,num_patch,2048))

    model = load_model('./models/res-bi-sp_koniq.h5' )

    model_final = Model(inputs=model.input,outputs=model.layers[-3].output )
    model = build_model((n,num_patch,2048), model_final)
    model.load_weights('./models/konvid_2_bilstm_join.h5')

    videos_features = os.listdir('./features')

    scores = []
    names = []
    print('Predicting MOS ..   ')
    print('======================================================')
    for i in tqdm(videos_features):
        x_test = np.load('./features/' + i)
        x_test = x_test.reshape((1,n,num_patch,2048))
        y_pred = model.predict(x_test)
        y_pred = y_pred.tolist()
        na = i.split('.')[0] + '.mp4'
        names.append(na)
        scores.append(y_pred[0][0]*5)


    df = pd.DataFrame(list(zip(names, scores)),
               columns =['Videos', 'scores'])

    df.to_csv('predicted_mos.csv', index=False)
    print('======================================================')

    print('Done !')
    t3 = time.time() - t2
    print('Time needed to predict mos: ',convert(t3))

    print('======================================================')



