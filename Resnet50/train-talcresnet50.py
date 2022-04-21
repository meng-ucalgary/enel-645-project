#!/usr/bin/env python
# coding: utf-8

# In[52]:


import sys
sys.path.append(".")


# In[55]:


import numpy as np
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB1
import matplotlib.pylab as plt
from pathlib import Path
from PlayingCardsGenerator import CardsDataGenerator
from tensorflow.keras import layers


# In[40]:


model_name_it = "/home/mmylee/term-project/Outputs/Resnet50_Scratch.h5"

# In[41]:

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
else:
  print("No GPU device found")


# In[42]:


early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 20)


# In[43]:


monitor_it = tf.keras.callbacks.ModelCheckpoint(model_name_it, monitor='val_loss',                                             verbose=0,save_best_only=True,                                             save_weights_only=True,                                             mode='min')


# In[44]:


def scheduler(epoch, lr):
    if epoch%40 == 0 and epoch!= 0:
        lr = lr/2
    return lr


# In[45]:


lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose = 0)


# In[46]:


gen_params = {"featurewise_center":False,\
              "samplewise_center":False,\
              "featurewise_std_normalization":False,\
              "samplewise_std_normalization":False,\
              "rotation_range":90,\
              "width_shift_range":0.3,\
              "height_shift_range":0.3, \
              "shear_range":0.3, \
              "zoom_range":0.3,\
              "vertical_flip":True, \
              "brightness_range": (0.2, 2)}

# In[35]:


generator = CardsDataGenerator(**gen_params, validation_split=0.2,  preprocessing_function = tf.keras.applications.resnet50.preprocess_input)
generator = ImageDataGenerator(**gen_params, validation_split=0.2,  preprocessing_function = tf.keras.applications.efficientnet.preprocess_input)

# In[36]:


bs = 16 # batch size


# In[37]:


path = Path("/home/mmylee/term-project/dataset/")


# In[27]:


img_height = 224
img_width = 224


# In[28]:


classes_names = ["2_clubs","2_diamonds","2_hearts","2_spades",               "3_clubs","3_diamonds","3_hearts","3_spades",               "4_clubs","4_diamonds","4_hearts","4_spades",               "5_clubs","5_diamonds","5_hearts","5_spades",               "6_clubs","6_diamonds","6_hearts","6_spades",               "7_clubs","7_diamonds","7_hearts","7_spades",               "8_clubs","8_diamonds","8_hearts","8_spades",               "9_clubs","9_diamonds","9_hearts","9_spades",               "10_clubs","10_diamonds","10_hearts","10_spades",               "ace_clubs","ace_diamonds","ace_hearts","ace_spades",               "jack_clubs","jack_diamonds","jack_hearts","jack_spades",               "king_clubs","king_diamonds","king_hearts","king_spades",               "queen_clubs","queen_diamonds","queen_hearts","queen_spades"]


# In[48]:


train_generator = generator.flow_from_directory(
    directory = path,
    target_size=(img_height, img_width),
    batch_size=bs,
    class_mode="categorical",
    subset='training',
    interpolation="nearest",
    classes=classes_names) # set as training data


# In[49]:


validation_generator = generator.flow_from_directory(
    directory = path,
    target_size=(img_height, img_width),
    batch_size=bs,
    class_mode="categorical",
    subset='validation',
    interpolation="nearest",
    classes=classes_names) # set as validation data


# In[58]:


#Exploratory data analysis


Xbatch, Ybatch = validation_generator.__getitem__(0)




# In[51]:


# Defining the model

trainable_flag = True
include_top_flag = False
weigths_value = 'imagenet'

if trainable_flag:
    include_top_flag = True
    weigths_value = None
else:
    include_top_flag = False
    weigths_value = 'imagenet'


# In[32]:


print(weigths_value)
print(include_top_flag)
print(trainable_flag)


base_model = tf.keras.applications.resnet50.ResNet50(
    weights=weigths_value,
    input_tensor=None,

    pooling=None,
    
    input_shape=(img_height, img_width, 3),
    include_top=include_top_flag,
    classes=len(classes_names) )
base_model.trainable = trainable_flag
inputs = layers.Input(shape=(img_height,img_width,3))
outputs = base_model(inputs)
model = tf.keras.Model( inputs,  outputs)

# In[33]:





# In[26]:




# In[22]:


print("Initial Training Model")
print(model.summary())


# In[5]:


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001), #1e-4
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[86]:



history_it = model.fit(train_generator, epochs=1000, verbose = 1,                        workers=8, validation_data = (validation_generator),  callbacks= [monitor_it,early_stop, lr_schedule])


# In[ ]:


model.save_weights('/home/mmylee/term-project/Outputs/Resnet50_Scratch.h5')
np.save('/home/mmylee/term-project/Outputs/Resnet50_Scratch52_history.npy',history_it.history)
