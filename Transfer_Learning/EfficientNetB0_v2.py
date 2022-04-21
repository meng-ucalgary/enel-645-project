"""
# Author
    Bhavyai Gupta

# Purpose
    Transfer Learning with EfficientNetB0 models using pretrained imagenet weights

# Model summary:
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #
    =================================================================
    input_1 (InputLayer)         [(None, 224, 224, 3)]     0
    _________________________________________________________________
    efficientnetb0 (Functional)  (None, 7, 7, 1280)        4049571
    _________________________________________________________________
    avg_pool (GlobalAveragePooli (None, 1280)              0
    _________________________________________________________________
    batch_normalization (BatchNo (None, 1280)              5120
    _________________________________________________________________
    top_dropout (Dropout)        (None, 1280)              0
    _________________________________________________________________
    flatten (Flatten)            (None, 1280)              0
    _________________________________________________________________
    prediction (Dense)           (None, 52)                66612
    =================================================================
    Total params: 4,121,303
    Trainable params: 69,172
    Non-trainable params: 4,052,131
    _________________________________________________________________
"""

# #############################################################################
# Imports
# #############################################################################
import sys
from pathlib import Path
import pickle
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# #############################################################################
# TALC configuration
# #############################################################################
sys.path.append('.')
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


# #############################################################################
# Define input image size and output class names
#
#  Base model       resolution
# -----------------+-------------
#  EfficientNetB0  | 224
#  EfficientNetB1  | 240
#  EfficientNetB2  | 260
#  EfficientNetB3  | 300
#  EfficientNetB4  | 380
#  EfficientNetB5  | 456
#  EfficientNetB6  | 528
#  EfficientNetB7  | 600
# #############################################################################
img_h = 224
img_w = 224

class_names = ['2_clubs', '2_diamonds', '2_hearts', '2_spades',
               '3_clubs', '3_diamonds', '3_hearts', '3_spades',
               '4_clubs', '4_diamonds', '4_hearts', '4_spades',
               '5_clubs', '5_diamonds', '5_hearts', '5_spades',
               '6_clubs', '6_diamonds', '6_hearts', '6_spades',
               '7_clubs', '7_diamonds', '7_hearts', '7_spades',
               '8_clubs', '8_diamonds', '8_hearts', '8_spades',
               '9_clubs', '9_diamonds', '9_hearts', '9_spades',
               '10_clubs', '10_diamonds', '10_hearts', '10_spades',
               'ace_clubs', 'ace_diamonds', 'ace_hearts', 'ace_spades',
               'jack_clubs', 'jack_diamonds', 'jack_hearts', 'jack_spades',
               'king_clubs', 'king_diamonds', 'king_hearts', 'king_spades',
               'queen_clubs', 'queen_diamonds', 'queen_hearts', 'queen_spades']

num_classes = len(class_names)

dataset_path = Path('dataset/')


# #############################################################################
# Define your callbacks
# #############################################################################
model_name_it = 'Outputs/TL_EfficientNetB0_v2_it.h5'
model_name_ft = 'Outputs/TL_EfficientNetB0_v2_ft.h5'

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

monitor_it = tf.keras.callbacks.ModelCheckpoint(model_name_it,
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=False,
                                                mode='min')

monitor_ft = tf.keras.callbacks.ModelCheckpoint(model_name_ft,
                                                monitor='val_loss',
                                                verbose=1,
                                                save_best_only=True,
                                                save_weights_only=False,
                                                mode='min')


def scheduler(epoch, lr):
    if epoch % 30 == 0 and epoch != 0:
        lr = lr/2
    return lr


lr_schedule = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)


# #############################################################################
# Keras data augmentation
# #############################################################################
gen_params = {'featurewise_center': False,
              'samplewise_center': False,
              'featurewise_std_normalization': False,
              'samplewise_std_normalization': False,
              'rotation_range': 180,
              'width_shift_range': 0.3,
              'height_shift_range': 0.3,
              'shear_range': 0.3,
              'zoom_range': 0.3,
              'vertical_flip': True,
              'brightness_range': (0.2, 2)}

batch_size = 16

generator = ImageDataGenerator(**gen_params,
                               validation_split=0.2,
                               preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)

# set as training data
train_gen = generator.flow_from_directory(directory=dataset_path,
                                          target_size=(img_h, img_w),
                                          batch_size=batch_size,
                                          classes=class_names,
                                          class_mode='categorical',
                                          subset='training',
                                          shuffle=True,
                                          interpolation='nearest',
                                          seed=42)

# set as validation data
validation_gen = generator.flow_from_directory(directory=dataset_path,
                                               target_size=(img_h, img_w),
                                               batch_size=batch_size,
                                               classes=class_names,
                                               class_mode='categorical',
                                               subset='validation',
                                               interpolation='nearest',
                                               seed=42)


# #############################################################################
# Transfer Learning
#
# Choosing pretrained model without the top, with frozen layers
# Then add a top
# #############################################################################
base_model = EfficientNetB0(include_top=False,
                            weights='imagenet',
                            input_shape=(img_h, img_w, 3))

# freeze the pre-trained weights
base_model.trainable = False

# rebuild top
x = base_model(base_model.input, training=False)
x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dropout(0.2, name='top_dropout')(x)
x = tf.keras.layers.Flatten()(x)
out = tf.keras.layers.Dense(
    num_classes, activation='softmax', name='prediction')(x)

# compile
model = tf.keras.Model(inputs=base_model.input,
                       outputs=out,
                       name='EfficientNetB0')

print('\n\n\nInitial training model\n')
print(model.summary())


# #############################################################################
# Training the model
# #############################################################################
adam_optimizer_it = tf.keras.optimizers.Adam(learning_rate=1e-3)

model.compile(optimizer=adam_optimizer_it,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_it = model.fit(train_gen,
                       epochs=100,
                       verbose=1,
                       workers=8,
                       callbacks=[early_stop, monitor_it, lr_schedule],
                       validation_data=(validation_gen))


# #############################################################################
# Saving the model history
# #############################################################################
it_file = open("Outputs/TL_EfficientNetB0_v2_it_history.pkl", "wb")
pickle.dump(history_it.history, it_file)
it_file.close()


# #############################################################################
# Fine tuning the model
# #############################################################################
model = tf.keras.models.load_model(model_name_it)
model.trainable = True

print('\n\n\nFine tuning model\n')
print(model.summary())

adam_optimizer_ft = tf.keras.optimizers.Adam(learning_rate=1e-8)

model.compile(optimizer=adam_optimizer_ft,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_ft = model.fit(train_gen,
                       epochs=50,
                       verbose=1,
                       workers=8,
                       callbacks=[early_stop, monitor_ft, lr_schedule],
                       validation_data=(validation_gen))


# #############################################################################
# Saving the model history
# #############################################################################
ft_file = open("Outputs/TL_EfficientNetB0_v2_ft_history.pkl", "wb")
pickle.dump(history_ft.history, ft_file)
ft_file.close()
