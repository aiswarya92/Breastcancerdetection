from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import tensorflow as tf

trainData = 'dataset/train'
testData = 'dataset/test'
dnnModel = 'dnn_model.h5'

imgW = 50
imgH = 50
trainSamples = 500
testSamples = 5
epochs = 100
batchSize = 10

def createModel(input_shape):
    model = Sequential()
    
    model.add(Flatten(input_shape = input_shape))
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('tanh'))
    
    model.compile(optimizer = 'rmsprop', 
                  loss = 'CategoricalCrossentropy',
                  metrics = ['accuracy'])
    
    return model

def train_model():
        
    datagen = ImageDataGenerator(rescale=1./255)
    
    model = createModel((50,50,3))
    print(model)
    
    train_generator = datagen.flow_from_directory(
            trainData,
            target_size = (imgW, imgH),
            batch_size = batchSize,
            shuffle = False)
    
    test_generator = datagen.flow_from_directory(
            testData,
            target_size = (imgW, imgH),
            batch_size = batchSize,
            shuffle = False)
    
    model.fit_generator(
            train_generator,
            steps_per_epoch = trainSamples, 
            epochs = epochs,
            validation_data = test_generator,
            validation_steps = trainSamples
            )
    
    model.save_weights(dnnModel)
    
train_model()
