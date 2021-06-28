import numpy as np
import cv2
import os
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
from tensorflow.keras.layers import Dense ,Dropout ,Input,concatenate,Conv2D,Reshape,GlobalMaxPooling2D,Flatten,GlobalAveragePooling2D
import argparse
from tqdm import tqdm
from utils.utils import *



tf.keras.backend.clear_session()





class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size=1, patches = 15,
                  shuffle=False, list_IDs='',overlapping = 0.2):
        'Initialization'
        self.batch_size = batch_size
        self.patches = patches
        self.shuffle = shuffle
        self.list_IDs = list_IDs
        self.overlapping = overlapping
        self.on_epoch_end()
   
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs)/ self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of IDs
        batch = [self.list_IDs[k] for k in indexes]
        
        # Generate data
        name, X, y = self.__data_generation(batch)

        return name, X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, batch):

        # Initialization
        X = np.empty((self.batch_size, self.patches,224,224, 3))
        
        y = np.empty((self.batch_size), dtype=np.float32)

        
        # Generate data
        for i, ID in enumerate(batch):
            imgs = crop_image(ID[0],overlapping= self.overlapping, num_patch= self.patches)
            name = ID[0].split('/')[-1]
            y[i] = ID[1]
            imgs = np.array(imgs)
            for j in range(self.patches):
              X[i,j,:,:,:]=imgs[j,:,:,:]
              X[i,j,:,:,:] = keras.applications.resnet50.preprocess_input(X[i,j,:,:,:])

            X = np.reshape(X,(self.patches,224,224,3))
            
                   
              
        return name, X, y



def extract_feaures(model,list_IDs, samples,num_patch ,overlapping,batch_size=1):
	images = DataGenerator(batch_size=batch_size, list_IDs=list_IDs, patches = num_patch, overlapping = overlapping)
	name = []
	features_X = np.zeros((samples,2048*num_patch))
	features_Y = np.zeros((samples,))
	i=0 
	for ID,X,Y in tqdm(images):
		features = model.predict(X)
		features_X[i] = np.reshape(features,(2048*num_patch,))
		features_Y[i] = Y
		name.append(ID)

		if i == samples-1 : 
			return name,features_X,features_Y
		i+=1




if __name__ == '__main__':

	parser = argparse.ArgumentParser("features_extracion")

	parser.add_argument(
        '--frame_dir',
        default='',
        type=str,
        help='Directory path of frames')



	parser.add_argument(
        '--csv_file',
        default='',
        type=str,
        help='images metadata list csv file'
    )



	parser.add_argument(
        '--overlapping', 
        default=0.2, 
        type=float,
        help="overlapping between batches ( between 0 and 1).")


	parser.add_argument(
        '--num_patch',
        default=15,
        type=int,
        help='Number of cropped patches per frames.'
    )



	args = parser.parse_args()


	image_dir = args.frame_dir



	if not os.path.exists('./features'):
		os.makedirs('./features')

	train_list, val_list = prepare_datalist(path_to_csv = args.csv_file , images_dir= image_dir)



	num_patch = args.num_patch
	overlap = args.overlapping

	batch_shapes = (num_patch,224,224,3)

	model_final = model_build(batch_shapes)


	name,X,Y = extract_feaures(model_final,train_list,samples =len(train_list), batch_size=1,  num_patch = num_patch,overlapping= overlap)
	np.save('./features/x_train_konvid_' + str(num_patch) + '.npy',X)
	np.save('./features/y_train_konvid_' + str(num_patch) + '.npy',Y)

	name,X,Y = extract_feaures(model_final,val_list,samples =len(val_list), batch_size=1,  num_patch = num_patch,overlapping= overlap)
	np.save('./features/x_test_konvid_' + str(num_patch) + '.npy',X)
	np.save('./features/y_test_konvid_' + str(num_patch) + '.npy',Y)
	
	
	

