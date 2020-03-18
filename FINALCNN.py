import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
import os
from matplotlib.image import imread
import pandas as pd
print("Imports Successful")
train_size = 5760
test_size = 1440

batch_size = 60
num_classes = 3
epochs = 10

img_rows, img_cols = 1170, 580

count = 0
TrainArray = []
for file in os.listdir("C:/Users/aeshon/Desktop/Data/DataAndLabels/CroppedTrainData"):
    if len(TrainArray) >= 3840:
      count = 0
      while count < 1920:
        img = imread("C:/Users/aeshon/Desktop/Data/DataAndLabels/CroppedTrainData/Road " + str(count) + ".jpg")
        #print("Hi"+str(count))
        new_img = img[:,:,0]
        TrainArray.append(new_img)
        count = count + 1
      break

    if len(TrainArray) >= 1920:
      count = 0
      while count < 1920:
        img = imread("C:/Users/aeshon/Desktop/Data/DataAndLabels/CroppedTrainData/Water " + str(count) + ".jpg")
        #print("Hi"+str(count))
        new_img = img[:,:,0]
        TrainArray.append(new_img)
        count = count + 1
      continue
    
    img = imread("C:/Users/aeshon/Desktop/Data/DataAndLabels/CroppedTrainData/Gravel " + str(count) + ".jpg")
    #print("Hi"+str(count))
    new_img = img[:,:,0]
    TrainArray.append(new_img)
    count = count + 1
print("Train Array Synthesis Complete")
x_train = np.asarray(TrainArray)
del TrainArray
print("Array Deleted!")
count = 0
TestArray = []
for file in os.listdir("C:/Users/aeshon/Desktop/Data/DataAndLabels/CroppedTestData"):
    if len(TestArray) >= 960:
      count = 0
      while count < 480:
        img = imread("C:/Users/aeshon/Desktop/Data/DataAndLabels/CroppedTestData/Road " + str(count) + ".jpg")
        #print("Hi"+str(count))
        new_img = img[:,:,0]
        TestArray.append(new_img)
        count = count + 1
      break
    
    if len(TestArray) >= 480:
      count = 0
      while count < 480:
        img = imread("C:/Users/aeshon/Desktop/Data/DataAndLabels/CroppedTestData/Water " + str(count) + ".jpg")
        #print("Hi"+str(count))
        new_img = img[:,:,0]
        TestArray.append(new_img)
        count = count + 1
      continue
    
    img = imread("C:/Users/aeshon/Desktop/Data/DataAndLabels/CroppedTestData/Gravel " + str(count) + ".jpg")
    #print("Hi"+str(count))
    new_img = img[:,:,0]
    TestArray.append(new_img)
    count = count + 1
print("Test Array Synthesis Complete")
x_test = np.asarray(TestArray)
del TestArray
print("Array Deleted!")

x_train = x_train.reshape(train_size, 1170, 580, 1)
x_test = x_test.reshape(test_size, 1170, 580, 1)

print("x_train shape:", x_train.shape)
#print(x_train.shape[0], 'train samples')
#print(x_test.shape[0], 'test samples')

TrainLabels = np.asarray(pd.read_csv('C:/Users/aeshon/Desktop/Data/DataAndLabels/TrainingLabelsCompressed.csv'))

for i in range(len(TrainLabels)):
    TrainLabels[i] = int(TrainLabels[i])
    
y_train = TrainLabels

TestLabels = np.asarray(pd.read_csv('C:/Users/aeshon/Desktop/Data/DataAndLabels/TestingLabelsCompressed.csv'))

for i in range(len(TestLabels)):
    TestLabels[i] = int(TestLabels[i])
    
y_test = TestLabels

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("Label formatting complete!")

print('Data Configuration Successful! Moving on to model compilation')

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=(1170,580,1)))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                activation='relu',))
model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

print("Starting Model Compilation!")
model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.SGD(lr = 0.00001),
              metrics = ['accuracy'])

print("Model Compiled!")

model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs,
          verbose = 1, validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test, verbose = 1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("C:/Users/aeshon/Desktop/model.h5")
print("Saved model to disk")