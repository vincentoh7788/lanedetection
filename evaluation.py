import cv2
import numpy as np
import pickle
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import load_model
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, Activation, Dropout,UpSampling2D,Conv2DTranspose,Conv2D
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from PIL import Image as im

train_images = pickle.load(open("full_CNN_train.p", "rb"))

# Load image labels
labels = pickle.load(open("full_CNN_labels.p", "rb"))

# Make into arrays as the neural network wants these
train_images = np.array(train_images)
labels = np.array(labels)

# Normalize labels - training images get normalized to start in the network
labels = labels / 255

# Shuffle images along with their labels, then split into training/validation sets
train_images, labels = shuffle(train_images, labels)
# Test size may be 10% or 20%
X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1)
print("Train images:",X_train.shape)
print("Train labels:",y_train.shape)
print("Test images:",X_val.shape)
print("Test labels:",y_val.shape)

model = load_model('full_CNN_model.h5')

predictions = model.predict(X_val)
sample_input = X_val[0]
sample_output = predictions[0]*255
sample_output = np.squeeze(sample_output)
img_x = im.fromarray(sample_input)
img_y = im.fromarray(sample_output)
img_x.show()
img_y.show()
y_val_flat = y_val.reshape(-1, y_val.shape[-1])
predictions_flat = predictions.reshape(-1, predictions.shape[-1])

# Calculate mean squared error
mse = mean_squared_error(y_val_flat, predictions_flat)
print("Mean Squared Error on Test Set: .%4f" % mse)
