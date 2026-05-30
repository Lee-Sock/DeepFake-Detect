import json
import os
from distutils.dir_util import copy_tree
import shutil
import pandas as pd
import math

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import backend as K
print('TensorFlow version: ', tf.__version__)

# Set to force CPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#if tf.test.gpu_device_name():
#    print('GPU found')
#else:
#    print("No GPU found")

dataset_path = '.\\split_dataset\\'

tmp_debug_path = '.\\tmp_debug'
print('Creating Directory: ' + tmp_debug_path)
os.makedirs(tmp_debug_path, exist_ok=True)

def get_filename_only(file_path):
    file_basename = os.path.basename(file_path)
    filename_only = file_basename.split('.')[0]
    return filename_only

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import applications
from efficientnet.tfkeras import EfficientNetB0 #EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model, model_from_json

input_size = 128
batch_size_num = 32
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
test_path = os.path.join(dataset_path, 'test')
checkpoint_filepath = '.\\tmp_checkpoint'

test_datagen = ImageDataGenerator(
    rescale = 1/255    #rescale the tensor values to [0,1]
)

test_generator = test_datagen.flow_from_directory(
    directory = test_path,
    classes=['real', 'fake'],
    target_size = (input_size, input_size),
    color_mode = "rgb",
    class_mode = None,
    batch_size = 1,
    shuffle = False
)

'''
efficient_net = EfficientNetB0(
    weights = 'imagenet',
    input_shape = (input_size, input_size, 3),
    include_top = False,
    pooling = 'max'
)


best_model = Sequential()
best_model.add(efficient_net)
#best_model.add(Input(shape=(None, 128, 128, 3)))
best_model.add(Dense(units = 512, activation = 'relu'))
best_model.add(Dropout(0.5))
best_model.add(Dense(units = 128, activation = 'relu'))
best_model.add(Dense(units = 1, activation = 'sigmoid'))
#best_model.summary()

# Compile model
best_model.compile(optimizer = Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

#build the modal
#best_model.build()


# load the saved model
best_model = load_model(
    os.path.join(checkpoint_filepath, 'saved_model.keras'),
    safe_mode=False,
    compile = True
)
'''

#load json model
with open('json_model.json', 'r') as json_file:
    loaded_model_json = json_file.read()
best_model = tf.keras.models.model_from_json(loaded_model_json)

#load saved weights
best_model.load_weights(os.path.join(checkpoint_filepath, 'saved_weights.weights.h5'))

test_generator.reset()

preds = best_model.predict(
    test_generator,
    verbose = 1
)

test_results = pd.DataFrame({
    "Filename": test_generator.filenames,
    "Prediction": preds.flatten()
})
print(test_results)

#write out file with results
text_file = open("Output.txt", "w")
text_file.write(test_results.to_string())
text_file.close()


'''
#using exported artifact
reloaded_artifact = tf.saved_model.load(checkpoint_filepath)
predictions = reloaded_artifact.serve(test_generator)
print(predictions)
'''