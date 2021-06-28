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
from tensorflow.keras.layers import Dense ,Dropout ,Input,concatenate,Conv2D,Reshape,GlobalAveragePooling2D,Flatten
import argparse
from tqdm import tqdm

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


def crop_image(path, overlapping,num_patch):
	img = image.load_img(path)
	img = image.img_to_array(img)
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


def prepare_datalist(path_to_csv, images_dir):
	data1 = pd.read_csv(path_to_csv)
	li = data1.values.tolist()
	len_li = len(li)
	len_train = int((len_li/100)*80)
	len_test = len_li - len_train
	for i in range(len(li)):
		li[i][0]= str(int(li[i][0]))



	fr = []
	sc = []
	frames = os.listdir('./frames')
	len_frames = len(frames)
	nb_frames = len(frames) / len_li
	for frame in frames:
	  for l in li :
	    name = frame.split('.')[0]
	    if l[0] in name:
	      fr.append(frame)
	      sc.append(l[1])
	      break

	df = pd.DataFrame(list(zip(fr, sc)),
               columns =['name', 'score'])
	      
	li_p = df.values.tolist()
	for i in range(len(li_p)):
		li_p[i][0] = int(li_p[i][0].split('.png')[0])
		li_p[i][1] = li_p[i][1]/5

	li_p.sort()

	for i in li_p:
		i[0] = './frames/' + str(i[0]) + '.png'
 
	train = li_p[0:int(len_train*nb_frames)]
	test = li_p[int(len_train*nb_frames):]

	return(train, test)


def model_build(batch_shape):

	In = Input(batch_shape)
	model = ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))

	out1 = GlobalAveragePooling2D()(model.output)


	model_final = Model(inputs=model.input,outputs=out1 )

	

	for layer in model_final.layers:
		layer.trainable = False


	return model_final



