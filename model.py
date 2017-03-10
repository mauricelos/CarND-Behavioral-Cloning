from sklearn.utils import shuffle
import numpy as np
from skimage import io, transform
import os
import pandas
from keras.models import Sequential
from keras.layers import Cropping2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

dir = 'data/IMG/'
csv = 'data/driving_log.csv'

images = []
angles = []
dataframe = pandas.read_csv(csv, header=None)
dataset = dataframe.values
images_left = dataset[1:,1]
images_right = dataset[1:,2]
images_center = dataset[1:,0]
steering_angles = dataset[1:,3]

ret = np.cumsum(steering_angles, dtype=float)  #smoothing steering angles
ret[5:] = ret[5:] - ret[:-5]
steering_angles_ma = ret[5 - 1:] / 5

for right, left, center, angle in zip(images_right, images_left, images_center, steering_angles_ma):
    
        path, center_file = os.path.split(center)
        path, left_file = os.path.split(left)
        path, right_file = os.path.split(right)
                
        if np.isclose(angle, 0, 0.001):  #disinclude all angles around 0
            continue

        if angle > 0.95 or angle < -0.95:   #disinclude all angles harsher than 0.95/-0.95
            continue
        
        offset = 0.2   #applying offset to left und right angles
        
        left_angle = angle + offset
        
        right_angle = angle - offset
        
        images.append(transform.resize(io.imread(dir + center_file), (80, 160)))  #resizing images
        #images.append(np.fliplr(transform.resize(io.imread(dir + center_file), (80, 160))))  #flipping images if necessary
        angles.append(angle)
        
        images.append(transform.resize(io.imread(dir + "/" + left_file), (80, 160)))
        
        angles.append(left_angle)
        
        images.append(transform.resize(io.imread(dir + "/" + right_file), (80, 160)))
        
        angles.append(right_angle)
        #angles_reverted.append(angle)     #old .append from when I used flipped images

plt.hist(angles, bins= 100)  #to show distribution of angles
plt.title("Distribution")
plt.xlabel('angles')
plt.ylabel('amounts')
plt.plot()
        
X_train = np.array(images, dtype='float32')
y_train = np.array(angles, dtype='float32')

X_train, y_train = shuffle(X_train, y_train)

train_datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.02, fill_mode='nearest') #augment images

train_generator = train_datagen.flow(X_train, y_train, batch_size=128)

valid_datagen = ImageDataGenerator()

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size= 0.2, random_state=0) #splitting data

validation_generator = valid_datagen.flow(X_val, y_val, batch_size=128)


model = Sequential()

model.add(Cropping2D(cropping=((24,10), (0,0)), input_shape=(80, 160,3)))   #cropping images

model.add(Convolution2D(24, 5, 5, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(36, 5, 5, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(48, 5, 5, border_mode='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.1))

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.1))

model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, init='normal'))

model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

model.fit_generator(train_generator, samples_per_epoch=20016, nb_epoch=5, validation_data=validation_generator, nb_val_samples=2000)

model.save("model.h5")

print("Model saved")











