
# coding: utf-8

import os
import cv2
import random
import time

import wget
from wget import bar_thermometer

import tarfile
import pickle
import numpy as np

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Concatenate, Input, Merge
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D

import sklearn
import sklearn.linear_model
import sklearn.svm as svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, accuracy_score, auc, roc_curve, roc_auc_score
from sklearn.externals import joblib

from scipy.ndimage import uniform_filter

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

###################### Start of functions definitions ############################
def load_cifar_10():
    """
    Loading and preparing cifar 10 dataset
    
    Returns:
    x_train, y_train, x_test, y_test
    """
    dicts, test_file, meta = download_and_load_cifar_10()
    
    x_train, y_train = merge_sets(dicts)
    
    x_test, y_test = convert_images_from_batch(test_file)
    
    return x_train, y_train, x_test, y_test


def download_and_load_cifar_10():
    """
    Downloading and extraction cifar10 dataset
    
    Returns:
    dicts - list of loaded data batches (one batch contains 10000 images) from  cifar10 dataset
    test_file - test batch from cifar10 dataset
    meta - file contains information about classes in dataset
    """
    data_folder = 'data'
    extraction_folder = 'data/extraction'
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    
    filepath = download_file_from_url(url, data_folder)
    extracted_files = extract_files(filepath, extraction_folder)
    extracted_folder = find_folder(extracted_files, extraction_folder)
    dicts, test_file, meta  = unpickle_all(extracted_folder)
    return dicts, test_file, meta


def download_file_from_url(url, data_folder='.'):
    """
    Downloading dataset if needed

    url - url to dataset
    data_folder - path to folder where we choose to download data, default is '.'

    Returns:
    path_to_file - downloaded file
    """
    filename = wget.filename_from_url(url)
    path_to_file = os.path.abspath(data_folder + os.sep + filename)
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    if not os.path.exists(path_to_file):
        print('Downloading data from', url, 'and saving as', path_to_file)
        wget.download(url, out = data_folder, bar=bar_thermometer)
        print('\n','Done')
    else:
        print('File', os.path.abspath(path_to_file), 'already exists.')
    return path_to_file    


def extract_files(filepath, extraction_folder='.'):
    """
    Extracting file with dataset if needed

    filepath - gzipped file to extract
    extraction_folder - path to folder where we choose to extract data, default is '.'

    Returns:
    filenames - extracted filenames
    """
    if not os.path.exists(extraction_folder):
        os.makedirs(extraction_folder)
        
    extraction_folder = os.path.abspath(extraction_folder)   
    filenames=''
    
    if (filepath.endswith("tar.gz")):
        tar = tarfile.open(filepath, "r:gz")
        filenames = tar.getmembers()
        if not os.path.exists(extraction_folder + os.sep + filenames[0].name):
            print('Extracting file', filepath, 'in folder', extraction_folder)
            tar.extractall(path=extraction_folder)
            tar.close()
            print('Extracting done')
        else:
            print('File', filepath,'already extracted in', extraction_folder)
            
    return filenames


def find_folder(extracted_files, extraction_folder='.'):
    """
    Find extracted folder 

    extracted_files - gzipped file to extract
    extraction_folder - path to folder where we extracted data, default is '.'

    Returns:
    absolute path to found folder
    """
    extracted_folder = ''
    for file in extracted_files:
        if(file.isdir()):
            extracted_folder = file.name
    return os.path.abspath(extraction_folder + os.sep + extracted_folder)


def unpickle(file):
    """
    Unpickle file
    function from: https://www.cs.toronto.edu/~kriz/cifar.html

    file - file to unpickle

    Returns:
    dict - dictionary with unpickled data
    """
    meta = ''
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def unpickle_all(path):
    """
    Unpickle all files

    path - path fo folder containins files to unpickle

    Returns:
    dicts - list of dictionaries with unpickled train data
    test - dictionary for test data
    meta - file contains information about classes in dataset
    """
    dicts = []
    meta = ''
    test = ''
    if(os.path.isdir(path)):
        for file in os.listdir(path):
            try:
                unpickled_file = unpickle(path + os.sep + file)
                if(file.endswith('.meta')):
                    meta = unpickled_file
                elif(file.find('test') == 0):
                    test = unpickled_file
                else:
                    dicts.append(unpickled_file)
            except pickle.UnpicklingError:
                print('Not pickle file', file)
    return dicts, test, meta


def convert_images(img):
    """
    Convert images from batch to list of images

    raw - images to convert

    Returns:
    images - converted images
    """
    # Width and height of each image.
    img_size = 32
    # Number of channels in each image, 3 channels: Red, Green, Blue.
    num_channels = 3

    # Reshape the array to 4-dimensions.
    images = img.reshape([-1, num_channels, img_size, img_size])
    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])
    
    return images

def process_images(imgs, normalization=True, to_gray=True):
    """
    Function for performin normalization and conversion to gray image
    
    normalization - boolean flag if perform normalization of images to have values in range [0,1], default True
    to_grey - boolean flag if perform scale to gray colored image
    
    Returns:
    imgs - grayed and/or normalized image
    """
    if(to_gray):
        r, g, b = imgs[:,:,:,0], imgs[:,:,:,1], imgs[:,:,:,2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        imgs = gray.reshape(imgs.shape[0], imgs.shape[1], imgs.shape[2], 1)
        
    if(normalization):
        imgs = np.array(imgs, dtype=float) / 255.0    
     
    return imgs
    
def convert_images_from_batch(data):
    """
    Convert all images from batch

    data - batch of  images to convert

    Returns:
    images - converted images from given batch
    cls  - labels for given batch
    """
    raw_images = data[b'data']
    cls = np.array(data[b'labels'])
    images = convert_images(raw_images)
    
    return images, cls

def merge_sets(dicts):
    """
    Merge sets from given list of loaded batches

    dicts - list of dictionaries with unpickled  data

    Returns:
    list_images - merged list of all images from given dicts
    list_cls  - labels for merged batches
    """
    list_images = [] 
    list_cls = []

    for i in range(0, len(dicts)):
        images, cls = convert_images_from_batch(dicts[i])
        images_length = images.shape[0]
        if(len(list_images)==0):
            list_images = np.zeros((len(dicts)*images_length, images.shape[1], images.shape[2], images.shape[3]))
            list_cls = np.zeros((len(dicts)*images_length))
        else:
            idx_start = i*images_length
            idx_end = (i+1)*images_length
            list_images[idx_start:idx_end] = images
            list_cls[idx_start:idx_end] = cls
            
    return list_images, list_cls


def random_images_from_classes(num_classes, num_samples, x_set, y_set):
    """
    Generate random images from given sets, each sample contains one class

    num_classes - number of classes to choose from
    num_samples - number of samples to choose from each class
    x_set - array of all images
    y_set - array of all classes

    Returns:
    images_list - choosen images in list 
    """
    images_list = np.zeros((num_classes, num_samples, x_set.shape[1], x_set.shape[2], x_set.shape[3]))
    for i in range(0, num_classes):
        the_chosen_ones = np.where(y_set==i)
        images = x_set[the_chosen_ones]
        images_list[i] = random.sample(list(images), num_samples)
    return images_list

def plot_random_images_from_classes(num_classes, num_samples, x_set, y_set): 
    """
    Plot generated random images from given sets, each sample contains one class
    
    num_classes - number of classes to choose from
    num_samples - number of samples to choose from each class
    x_set - array of all images
    y_set - array of all classes
    """
    images = random_images_from_classes(num_classes, num_samples, x_set, y_set)
    fig, axarr = plt.subplots(num_classes, num_samples)
    fig.set_figheight(num_classes*1.5)
    fig.set_figwidth(num_samples*1.5)
    for i in range(0,num_classes):
        for j in range(0,num_samples):
            subplot = axarr[i,j]
            subplot.axis('off')
            subplot.imshow(images[i][j])

    matplotlib.pyplot.show(block=True)
            
def save_model(model, filename='model'):
    """
    Save keras model to file as json for structure and h5 file for weights
    
    model - model to save 
    filename - filename for saved model
    """
    model_json = model.to_json()
    with open(filename+".json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(filename+".h5")
    print('Saved', filename, 'to disk')

def read_model(filename='model'):
    """
    Load keras model from file (json)
    
    filename - filename of saved model
    
    Returns: 
    loaded_model - loaded keras model
    """
    json_file = open(filename+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(filename+".h5")
    print('Loaded', filename ,'from disk')
    return loaded_model            

def create_hog_features(x_set):
    """
    Creation of hog features for given set of images
    
    x_set - set of images for extraction of hog features
    
    Returns:
    h_set - set of hog features
    """
    winSize = (8,8)
    blockSize = (8,8)
    blockStride = (8,8)
    cellSize = (4,4)
    nbins = 3
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 2
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins
                            , derivAperture, winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels)

    print('Creating hog features for set with shape', x_set.shape)
    table_width = 0 
    table_length = 0 

    i=0
    for image in x_set:
        image=image.astype(np.uint8)
        r_image = cv2.resize(image, None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(r_image, cv2.COLOR_BGR2GRAY)
        h_image = hog.compute(gray)
        if(table_width==0):
            table_width = h_image.shape[0]
            table_length = len(x_set)
            h_set = np.zeros((table_length, table_width))
        
        h_set[i] = h_image.reshape(table_width)
        progress = i*100/table_length
        if(progress % 5 == 0):
            print(progress,'%')
        i=i+1
        
    print('100%')
    
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(h_set)
    h_set = scaling.transform(h_set)
    
    return h_set   

def train_svm(x_train, y_train):
    """
    Training SVM 
    
    x_train - set of features
    y_train - labels
    
    Returns:
    clf - trained SVM classifier
    """
    clf = svm.SVC(verbose=True)
    clf.fit(x_train, y_train)
    
    return clf

def train_cnn_net(x_train, y_train, x_test, y_test):
    """
    Training CNN 
    
    x_train - set of features
    y_train - labels
    
    Returns:
    big_model - trained CNN
    """
    
    FILTERS_1=32
    FILTERS_2=64
    P_DROPOUT_1=0.1
    P_DROPOUT_2=0.2
    batch_size=10
    epochs=5
    submodels=[]

    for kw in (2, 3):    # kernel sizes
        submodel = Sequential()
        submodel.add(Conv2D(FILTERS_1, (kw, kw), padding='same',
                     input_shape=x_train.shape[1:]))
        submodel.add(Activation('relu'))
        submodel.add(Conv2D(FILTERS_1, (kw, kw)))
        submodel.add(Activation('relu'))
        submodel.add(MaxPooling2D(pool_size=(2, 2)))
        submodel.add(Dropout(P_DROPOUT_1))
        submodels.append(submodel) 

    big_model = Sequential()
    big_model.add(Merge(submodels, mode="concat"))

    big_model.add(Conv2D(FILTERS_2, (3, 3), padding='same'))
    big_model.add(Activation('relu'))
    big_model.add(Conv2D(FILTERS_2, (3, 3)))
    big_model.add(Activation('relu'))
    big_model.add(MaxPooling2D(pool_size=(2, 2)))
    big_model.add(Dropout(0.25))
    big_model.add(Activation('relu'))

    big_model.add(Flatten())

    big_model.add(Dense(num_classes))
    big_model.add(Activation('softmax'))

    print('Compiling model')
    big_model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    start = time.time()
    big_model.fit([x_train, x_train], 
                  y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=([x_test, x_test], y_test),
                  shuffle=True)
    end = time.time()
    print('CNN trained in time:', end-start)
    
    return big_model

def test_cnn_net_model(model, x_test, y_test):
    """
    Testing CNN
    
    model - model for testing
    x_test - set of test features
    y_test- labels
    
    Returns:
    y_predicted - probabilities of labels returned by network
    scores - loss and accuracy
    auc_current - mean auc 
    """
    print("Test CNN!")
    start = time.time()
    y_predicted = model.predict([x_test, x_test])
    scores = model.evaluate([x_test, x_test], y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    auc_current = roc_auc_score(y_test, y_predicted)
    print('AUC:', auc_current)
    print('Total time scoring',time.time()-start)
    return y_predicted, scores, auc_current

def test_svm_model(clf, x_test, y_test):
    """
    Testing SVM (best score so far 0.47539999999999999)
    
    clf - classifier for testing
    x_test - set of test features
    y_test- labels
    
    Returns:
    y_predicted - labels predicted by SVM
    scores - accuracy
    """
    print("Test SVM!")
    start = time.time()
    score = clf.score(x_test, y_test)
    #prediction is slow for this case
    #y_predicted = clf.predict(x_test)
    print('Total time scoring',time.time()-start)
    return score

def convert_cnn_predictions_for_categories(y_predicted):
    """
    Conversion of predicted probabilities of labels returned by network for most probably labels
    
    y_predicted - probabilities to convert (multilabel)
    
    Returns:
    y_predicted_categories - most probably labels
    """
    y_predicted_categories = np.zeros((len(y_predicted))).astype(int)

    i=0
    for row in y_predicted:
        max_value = row.argmax()
        y_predicted_categories[i] = max_value.astype(int)
        i=i+1
    
    return y_predicted_categories

###################### End of functions definitions ############################

#Main script

x_train, y_train, x_test, y_test = load_cifar_10()

load_best_model= True

answer = input("Load last best trained models of CNN? [Y/n]")
if(answer=='n' or answer=='N'):
    print("OK! We will train our convolutional net again! And again... and again... \nThis can take a while...")
    load_best_model = False
else:
    print("Great choice! You will save time and test last trained model!")

num_classes = 10
num_samples = 10

if(load_best_model==False):
    #Preparation of train data for CNN
    cnn_y_train = keras.utils.to_categorical(y_train, num_classes)
    cnn_x_train = process_images(x_train, normalization=True, to_gray=False)

#Preparation of data for CNN
cnn_x_test = process_images(x_test, normalization=True, to_gray=False)
cnn_y_test = keras.utils.to_categorical(y_test, num_classes)

if(load_best_model):
    cnn = read_model('best_model_long_train')
    cnn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
else:
    cnn = train_cnn_net(cnn_x_train, cnn_y_train, cnn_x_test, cnn_y_test) 
    answer = input("Save new model to file? [y/N]")
    if(answer=='y' or answer=='Y'):
        print("OK! Just give us the filename for saving cnn model (default is 'model')! ")
        filename=input('Type filename for cnn:')
        save_model(big_model, filename)    

answer = input("Load last best trained models of SVM? [Y/n]")
load_best_model = True
if(answer=='n' or answer=='N'):
    print("OK! We will train our SVM again! And again... and again... ")
    load_best_model = False
else:
    print("Great choice! You will save time and test last trained model!")

if(load_best_model):
    clf = joblib.load('simple_svm_04753.pkl')
else:
    #Preparation of train data for SVM classifier (not normalized, but hog features extracted)
    print(x_train.shape)
    svm_x_train = create_hog_features(x_train.copy())
    clf = train_svm(svm_x_train, svm_y_train)
    
    answer = input("Save new model? [y/N]")
    if(answer=='y' or answer=='Y'):
        print("OK! Just give us the filename! ")
        filename=input('Type filename for svm:')
        joblib.dump(clf, filename + '.pkl') 

print("Testing trained models!")
y_cnn_predicted_probs, cnn_scores, auc = test_cnn_net_model(cnn, cnn_x_test, cnn_y_test)
y_cnn_predicted = convert_cnn_predictions_for_categories(y_cnn_predicted_probs)

answer = input("Testing trained model of SVM can take a while, we can tell you what our last best score is! Test it anyway? [y/N]")
if(answer=='y' or answer=='Y'):
    #Preparation of test data for SVM classifier (not normalized, but hog features extracted)
    svm_x_test = create_hog_features(x_test.copy())
    svm_y_test = y_test
    svm_score = test_svm_model(clf, svm_x_test, svm_y_test)  
else:
    svm_score = 0.47539999999999999
      
print("CNN scores are",'loss:',cnn_scores[0],'accuracy:', cnn_scores[1])
print("SVM accuracy is", svm_score)


answer = 'n'
while(answer!='q' and answer!='Q'):
    print("Plot original images and labes!")
    plot_random_images_from_classes(num_classes, num_samples, process_images(x_test, normalization=True, to_gray=False), y_test)
    print("Plot predicted labels and images with CNN!")
    plot_random_images_from_classes(num_classes, num_samples, process_images(x_test, normalization=True, to_gray=False), y_cnn_predicted)
    answer = input("Press q/Q to quit or any key to plot other images")


