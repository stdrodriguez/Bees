#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#Common packages
import numpy as np
import pandas as pd
import warnings
import os

# ML
import scipy
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Charts
import matplotlib.pyplot as plt
import seaborn as sns

#Keras/Tensorflow
import tensorflow as tf
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D, Dropout, BatchNormalization,LeakyReLU
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau

# Image processing
import imageio
import skimage
import skimage.io
import skimage.transform
import tensorboard
import datetime

img_folder='./data/imgs/'

# One hot encoding

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
categories = {}
class_indices = {}

def class_columns(df) :
    return np.column_stack((np.asarray(df['subspecies']), np.asarray(df['health'])))

def setup_onehot(df) :
    # Fit one hot encoder
    enc.fit(class_columns(df))
    # Get categories
    categories['subspecies'] = enc.categories_[0]
    categories['health'] = enc.categories_[1]
    # Get indices
    class_indices['subspecies'] = np.arange(len(categories['subspecies']))
    class_indices['health'] = len(categories['subspecies']) + np.arange(len(categories['health']))

def onehot_encoding(df, class_name) :
    if categories == {} :
        raise ValueError('Run setup_onehot first')
    
    return enc.transform(class_columns(df)).toarray()[:,class_indices[class_name]]

def read_data() :
    bees=pd.read_csv('./data/bees_train.csv', 
                index_col=False,
                dtype={'subspecies':'category', 'health':'category','caste':'category'})
    bees_test_for_evaluation=pd.read_csv('./data/bees_test.csv', 
                index_col=False,  
                dtype={'caste':'category'})
    
    setup_onehot(bees)
 
    return bees, bees_test_for_evaluation

def read_img(file, img_folder, img_width, img_height, img_channels):
    """
    Read and resize img, adjust channels. 
    @param file: file name without full path
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.io.imread(img_folder + file)
        img = skimage.transform.resize(img, (img_width, img_height), mode='reflect', )
    return img[:,:,:img_channels]

def plot_images(data, attribute, samples) :
    if len(samples) < 2 or len(samples) > 5 : 
        raise ValueError('len(samples) must be in [2, 5]') 
        
    _, ax = plt.subplots(nrows = 1, ncols = len(samples), figsize = (20, 5))
    for i, img_idx in enumerate(samples) :
        attrname = data[attribute].iloc[img_idx]
        filename = './data/imgs/' + data['file'].iloc[img_idx]
        img = imageio.imread(filename)
        ax[i].imshow(img)
        ax[i].set_title(filename + ' : ' + attrname, fontsize = 12)
    plt.tight_layout()
    plt.show()

def value_counts(data, attribute) :
    data[attribute].value_counts().plot(kind = 'bar')
    plt.ylabel('Count')
    plt.title(attribute)
    plt.show()

def split(bees):
    """ 
    Split to train, test and validation. 
    
    @param bees: Total Bees dataset to balance and split
    @return:  train bees, validation bees, test bees
    """
    # Split to train and test before balancing
    train_bees, test_bees = train_test_split(bees, random_state=24)

    # Split train to train and validation datasets
    # Validation for use during learning
    train_bees, val_bees = train_test_split(train_bees, test_size=0.1, random_state=24)

    return(train_bees, val_bees, test_bees)
	
def load_images_and_target(train_bees, val_bees, test_bees, y_field_name, img_width, img_height, img_channels):
    """
    Load images for features, drop other columns
    One hot encode for label, drop other columns
    @return: train images, validation images, test images, train labels, validation labels, test labels
    """
    # Bees already splitted to train, validation and test
    # Load and transform images to have equal width/height/channels. 
    # Use np.stack to get NumPy array for CNN input
    
    # Train data
    train_X = np.stack(train_bees['file'].apply(lambda x: read_img(x, img_folder, img_width, img_height, img_channels)))
    train_y = pd.DataFrame(onehot_encoding(train_bees, y_field_name))
    # train_y  = pd.get_dummies(train_bees[y_field_name], drop_first=False)

    # Validation during training data to calc val_loss metric
    val_X = np.stack(val_bees['file'].apply(lambda x: read_img(x, img_folder, img_width, img_height, img_channels)))
    val_y = pd.DataFrame(onehot_encoding(val_bees, y_field_name))
    # val_y = pd.get_dummies(val_bees[y_field_name], drop_first=False)

    # Test data
    test_X = np.stack(test_bees['file'].apply(lambda x: read_img(x, img_folder, img_width, img_height, img_channels)))
    test_y = pd.DataFrame(onehot_encoding(test_bees, y_field_name))
    # test_y = pd.get_dummies(test_bees[y_field_name], drop_first=False)

    return (train_X, val_X, test_X, train_y, val_y, test_y)	



def class_weights(df, class_name) :
    # Hint: usar
    # http://scikit-learn.org/stable/modules/generated/sklearn.utils.class_weight.compute_class_weight.html
    y = df[class_name]
    return sklearn.utils.class_weight.compute_class_weight("balanced",categories[class_name],y)



def train(NumArq,Iteration,model,
			train_X,
			train_y, 
			batch_size,
			epochs,
			validation_data_X,
			validation_data_y,
			steps_per_epoch,
			rotation_range,  # randomly rotate images in the range (degrees, 0 to rotation_range)
			zoom_range, # Randomly zoom image 
			width_shift_range,  # randomly shift images horizontally (fraction of total width)
			height_shift_range,  # randomly shift images vertically (fraction of total height)
			horizontal_flip,  # randomly flip images
			vertical_flip,
			patience,
			class_weights, 
			learn_rate,
			StartTime,
			Optimizer):
				
	generator = ImageDataGenerator(
				featurewise_center=False,  # set input mean to 0 over the dataset
				samplewise_center=False,  # set each sample mean to 0
				featurewise_std_normalization=False,  # divide inputs by std of the dataset
				samplewise_std_normalization=False,  # divide each input by its std
				zca_whitening=True,  # apply ZCA whitening
				rotation_range=rotation_range,  # randomly rotate images in the range (degrees, 0 to rotation_range)
				zoom_range = zoom_range, # Randomly zoom image 
				width_shift_range=width_shift_range,  # randomly shift images horizontally (fraction of total width)
				height_shift_range=height_shift_range,  # randomly shift images vertically (fraction of total height)
				horizontal_flip=horizontal_flip,  # randomly flip images
				vertical_flip=vertical_flip)
				
    
	#StartTime = datetime.datetime.now()
	fileName = "TrainningLog.txt"
	file1 = open(fileName,"a")

	file1.write("Arquitectura,Exp,StarTime,BatchSize,Epochs,Lr,TestLoss,TestAccuracy,TrainigLoss,ValidationLoss,TrainingAcc,ValidationAcc\n") 
	file1.close()
    
	generator.fit(train_X)
	#Train
	##Callbacks
	earlystopper = EarlyStopping(monitor='val_loss', min_delta=0.005, patience=patience, verbose=1,restore_best_weights=True)
    
	prefix = str(NumArq) + "_" + str(Iteration) + "_"
	if (os.path.isdir("logs/")==False):
		os.mkdir("logs/")
        
	logdir = "logs/" + prefix  + StartTime.strftime("%Y%m%d-%H%M%S")
    
	tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    
    #Optimizer = 1 Adam, 2 SGD
	if (Optimizer == 1):
		opt = keras.optimizers.Adam(lr = learn_rate)
		print("Adam")
	else:
		opt = keras.optimizers.SGD(lr = learn_rate, momentum = 0.05)
		print("sgd")
        
	model.compile(loss = 'categorical_crossentropy', optimizer = opt,metrics = ['accuracy'])
    
	training = model.fit_generator(generator.flow(train_X,train_y, batch_size)
						,epochs=epochs
						,validation_data=[validation_data_X, validation_data_y]
						,steps_per_epoch=steps_per_epoch
						,callbacks=[tensorboard_callback, earlystopper]
						,class_weight = class_weights)

	scores = model.evaluate(validation_data_X, validation_data_y, verbose=1)
   
	file1 = open(fileName,"a")
	row =   "1," + str(Iteration) + ',' + StartTime.strftime("%Y%m%d-%H%M%S") + ","
	row = row + str(batch_size) + "," + str(epochs) + "," + str(learn_rate) + "," + str(scores[0])
	row = row + "," + str(scores[1]) + "," + str(min(training.history['loss'])) + "," + str(min(training.history['val_loss'])) + ","
	row = row + str(max(training.history['acc']) )  + "," + str(max(training.history['val_acc'])) + "\n"
    
	file1.write(row)
	file1.close()
	if os.path.isdir("Trained_Models/")==False:
		os.mkdir("Trained_Models/")

	model.save('Trained_Models/' + prefix + str(StartTime.strftime("%Y%m%d-%H%M%S")) + '.h5')
    
	return training, model

def eval_model(training, model, test_X, test_y, field_name):
    """
    Model evaluation: plots, classification report
    @param training: model training history
    @param model: trained model
    @param test_X: features 
    @param test_y: labels
    @param field_name: label name to display on plots
    """
    ## Trained model analysis and evaluation
    f, ax = plt.subplots(2,1, figsize=(5,5))
    ax[0].plot(training.history['loss'], label="Loss")
    ax[0].plot(training.history['val_loss'], label="Validation loss")
    ax[0].set_title('%s: loss' % field_name)
    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].legend()
    
    # Accuracy
    ax[1].plot(training.history['acc'], label="Accuracy")
    ax[1].plot(training.history['val_acc'], label="Validation accuracy")
    ax[1].set_title('%s: accuracy' % field_name)
    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()
    plt.tight_layout()
    plt.show()

    # Accuracy by category
    test_pred = model.predict(test_X)
    
    acc_by_category = np.logical_and((test_pred > 0.5), test_y).sum()/test_y.sum()
    acc_by_category.plot(kind='bar', title='Accuracy by %s' % field_name)
    plt.ylabel('Accuracy')
    plt.show()

    # Print metrics
    print("Classification report")
    test_pred = np.argmax(test_pred, axis=1)
    test_truth = np.argmax(test_y.values, axis=1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(metrics.classification_report(test_truth, test_pred, target_names=categories[field_name]))

    # Loss function and accuracy
    test_res = model.evaluate(test_X, test_y.values, verbose=0)
    print('Loss function: %s, accuracy:' % test_res[0], test_res[1])	