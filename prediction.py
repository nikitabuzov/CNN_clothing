import keras
import numpy as np
from keras.models import load_model
import cv2
import glob
import numpy as np
from tqdm import tqdm
import re
import os

# Load Testing data
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

cwd = os.getcwd()
path = cwd+'/hw4_test/'
for filename in os.listdir(path):
    num = filename[:-4]
    num = num.zfill(4)
    new_filename = num + ".png"
    os.rename(os.path.join(path, filename), os.path.join(path, new_filename))

predict = []
files = glob.glob ("hw4_test/*.png")
for myFile in tqdm(sorted(files, key=numericalSort)):
    image = cv2.imread (myFile)
    predict.append (image)

# convert the data to the right type
predict = np.array(predict,dtype='float32')
predict /= 255

# Load the trained Keras Model
model = load_model('model.h5')

# Run predictions
prediction = model.predict_classes(predict)

# Save the pridictions into txt file
np.savetxt('prediction.txt', prediction, delimiter='', fmt='%.1i')
