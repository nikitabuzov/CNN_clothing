import cv2
import glob
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle

#Train data
train = []
train_labels = []
files = glob.glob ("hw4_train/0/*.png") # your image path
for myFile in tqdm(files):
    image = cv2.imread (myFile)
    train.append (image)
    train_labels.append(0)

files = glob.glob ("hw4_train/1/*.png")
for myFile in tqdm(files):
    image = cv2.imread (myFile)
    train.append (image)
    train_labels.append(1)

files = glob.glob ("hw4_train/2/*.png")
for myFile in tqdm(files):
    image = cv2.imread (myFile)
    train.append (image)
    train_labels.append(2)

files = glob.glob ("hw4_train/3/*.png")
for myFile in tqdm(files):
    image = cv2.imread (myFile)
    train.append (image)
    train_labels.append(3)

files = glob.glob ("hw4_train/4/*.png")
for myFile in tqdm(files):
    image = cv2.imread (myFile)
    train.append (image)
    train_labels.append(4)

files = glob.glob ("hw4_train/5/*.png")
for myFile in tqdm(files):
    image = cv2.imread (myFile)
    train.append (image)
    train_labels.append(5)

files = glob.glob ("hw4_train/6/*.png")
for myFile in tqdm(files):
    image = cv2.imread (myFile)
    train.append (image)
    train_labels.append(6)

files = glob.glob ("hw4_train/7/*.png")
for myFile in tqdm(files):
    image = cv2.imread (myFile)
    train.append (image)
    train_labels.append(7)

files = glob.glob ("hw4_train/8/*.png")
for myFile in tqdm(files):
    image = cv2.imread (myFile)
    train.append (image)
    train_labels.append(8)

files = glob.glob ("hw4_train/9/*.png")
for myFile in tqdm(files):
    image = cv2.imread (myFile)
    train.append (image)
    train_labels.append(9)


train = np.array(train,dtype='float32') #as mnist
train_labels = np.array(train_labels,dtype='int') #as mnist
train, train_labels = shuffle(train, train_labels)
# save numpy array as .npy formats
np.save('train_data',train)
np.save('train_label',train_labels)
