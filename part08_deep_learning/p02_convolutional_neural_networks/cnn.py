# as the cnn data is huge in size..about 250 mb...it's not uploaded to github
# data download link is given below:
# 'https://sds-platform-private.s3-us-east-2.amazonaws.com/uploads/P14-Convolutional-Neural-Networks.zip'
# download the data and save as below as mine or change the path as you want to

import os

if os.name == 'posix':
    TRAIN_DATA = '/home/reyad/Codes/mlproject/keras_tf_conda/project/ml_a2z/part08_deep_learning/p02_convolutional_neural_networks/data/train/'
    TEST_DATA = '/home/reyad/Codes/mlproject/keras_tf_conda/project/ml_a2z/part08_deep_learning/p02_convolutional_neural_networks/data/test/'
else:
    TRAIN_DATA = 'G:/REYAD/CODES/mlproject/keras_tf/project/ml_a2z/part08_deep_learning/p02_convolutional_neural_networks/data/train/'
    TEST_DATA = 'G:/REYAD/CODES/mlproject/keras_tf/project/ml_a2z/part08_deep_learning/p02_convolutional_neural_networks/data/test/'


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# declaring model
classifier = Sequential()

# adding convolution layer
classifier.add(
        Conv2D(
                filters=32,
                kernel_size=(3, 3),
                input_shape=(64, 64, 3),
                activation='relu'
        )
)

# adding maxpooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# adding 2nd convolution layer
classifier.add(
        Conv2D(
                filters=32,
                kernel_size=(3, 3),
                activation='relu'
        )
)

# adding 2nd maxpooling layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# adding flatten layer
classifier.add(Flatten())

# full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# compiling model
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# fitting model to data
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(TRAIN_DATA,
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory(TEST_DATA,
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                         steps_per_epoch=8000/32,
                         epochs=10,
                         validation_data=test_set,
                         validation_steps=2000/16)
