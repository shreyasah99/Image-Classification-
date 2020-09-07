# Convolutional Neural Network

# Installing Keras
# Enter the following command in a terminal (or anaconda prompt for Windows users): conda install -c conda-forge keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))  # 32 = no of feature dectotor , no of feature map
                                                                                    #   (3,3) =no of rows and columns in feature deatector
           # 3 (channels) since we have coloured image r,g,b        # 64,64 dimension in each channel
           # order will be 64,64,3 since we use tensorflow backend
           
# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))       # size of subtable for taking maximum

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 128, activation = 'relu'))   # 128 = no of nodes in hidden layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))   # 1 = no of nodes in output layer
                # activation = sigmoid since only 2 possible output ; yes or no
                # if more than 2 possible outcomes use softmax activation function
                                                            
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# adam = stostic gradient descent      binary = since 2 possible outcomes  ; if more tahn 2 possible outcomes then categorical_crossentropy

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 25,
                         validation_data = test_set,
                         validation_steps = 2000)