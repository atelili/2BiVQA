"""
Usage:
    python features_extration -v (video directory) 
    -f (csv file name + mos) -o (overlapping between patches , default = 0.2) 
    -np (num patches, default=15) -nf (num frames, default=30)


Author : 
    Ahmed Telili
"""





import numpy as np
import cv2
import os 
from tensorflow.keras import Input
from tensorflow.keras import optimizers
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet169
from tensorflow.keras.models import Model
from sklearn.utils import shuffle
from tensorflow import keras
import pandas as pd
import csv
from tensorflow.keras.preprocessing import image
from tensorflow.keras import applications 


import PIL
import h5py
from PIL import Image
from keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.layers import Dense ,Dropout ,Input,concatenate,Conv2D,Reshape,GlobalMaxPooling2D,Flatten,GlobalAveragePooling2D
import argparse
import random
from tqdm import tqdm




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

def crop_image(img, overlapping,num_patch):

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

def TemporalCrop(input_video_path, nb):
	out = []
	final = []
	
	cap = cv2.VideoCapture(input_video_path)
	print(input_video_path)
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







class DataGenerator(keras.utils.Sequence):
    def __init__(self, batch_size=1, patches = 15,
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
        name, X, y = self.__data_generation(batch, nb)

        return name, X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, batch, nb=30):

        # Initialization
        X = np.empty((self.nb, self.patches,224,224, 3))
        
        y = np.empty((self.batch_size), dtype=np.float32)

        
        # Generate data
        for i, ID in enumerate(batch):

            imgs = TemporalCrop(ID[0], self.nb)

            for k in range(len(imgs)):
            	im = crop_image(imgs[k],overlapping= self.overlapping, num_patch= self.patches)
            	for j in range(self.patches):
            		im = np.array(im)
            		X[k,j,:,:,:]=im[j,:,:,:]
            		X[k,j,:,:,:] = keras.applications.resnet50.preprocess_input(X[k,j,:,:,:])
            name = ID[0].split('/')[-1]	
            y[i] = ID[1]	
		
                   
              
        return name, X, y



def prepare_datalist(path_to_csv, images_dir):
	data1 = pd.read_csv(path_to_csv)
	li = data1.values.tolist()
	for i in range(len(li)):
		li[i][0]= images_dir + '/' + str(li[i][0]).split('.')[0]+'.mp4'
	
	return(li)






def model_build(batch_shape):

	In = Input(batch_shape)
	model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

	out1 = layers.GlobalAveragePooling2D()(model.output)


	model_final = Model(inputs=model.input,outputs=out1 )

	

	for layer in model_final.layers:
		layer.trainable = False


	return model_final


def extract_feaures(model,list_IDs, samples, batch_size=1, num_patch = 25,overlapping= 0.2):
	images = DataGenerator(batch_size=batch_size, list_IDs=list_IDs, patches = num_patch, overlapping = overlapping)
	name = []
	features_X = np.zeros((samples,num_patch,2048))
	features_Y = np.zeros((1))
	i=0 
	for ID,X,Y in tqdm(images):
		for l in range(samples):
			features = model.predict(X[l,:,:,:,:])
			features_X[l,:,:] = features
			
      

		
		features_Y = Y
		ID = ID.split('.')[0]
		np.save('./features_X/'+ID,features_X)
		np.save('./features_y/'+ID,features_Y)



if __name__ == '__main__':

	parser = argparse.ArgumentParser("features_extraction")

	parser.add_argument('-v',
        '--video_dir',
        default='',
        type=str,
        help='Directory path of images')



	parser.add_argument('-f',
        '--csv_file',
        default='',
        type=str,
        help='images metadata list csv file'
    )



	parser.add_argument('-o',
        '--overlapping', 
        default=0.2, 
        type=float,
        help="overlapping between batches ( between 0 and 1).")


	parser.add_argument('-np',
        '--num_patch',
        default=25,
        type=int,
        help='Number of cropped patches per frames.'
    )
	parser.add_argument('-nf',
        '--num_frames',
        default=30,
        type=int,
        help='Number of cropped frames per video.'
    )



	args = parser.parse_args()


	image_dir = args.video_dir
	image_dir = os.path.expanduser(image_dir)

	if not os.path.exists('./features_X'):
		os.makedirs('./features_X')


	if not os.path.exists('./features_y'):
		os.makedirs('./features_y')

	li = prepare_datalist(path_to_csv = args.csv_file , images_dir= args.video_dir)


	



	num_patch = args.num_patch
	overlap = args.overlapping
	nb = args.num_frames

	batch_shapes = (num_patch,224,224,3)

	model_final = model_build(batch_shapes)


	extract_feaures(model_final,li,samples =nb, batch_size=1,  num_patch = num_patch,overlapping= overlap)
	


