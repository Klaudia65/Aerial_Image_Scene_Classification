import pathlib
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from MetricsCallback import MetricsCallback

train_data_dir = pathlib.Path('dataset/images_train_test_val/train')
image_count = len(list(train_data_dir.glob('*/*')))
print(f'Train images number: {image_count}')#7350

val_data_dir = pathlib.Path('dataset/images_train_test_val/validation')
image_count = len(list(val_data_dir.glob('*/*')))
print(f'Validation images number: {image_count}')#2100


#-------preprocessing------

batch_size = 32
image_height = 128 # the objects are a bit small

train_data = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_dir,
    image_size=(image_height, image_height),
    batch_size=batch_size,
    shuffle=True
)

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    val_data_dir,
    image_size=(image_height, image_height),
    batch_size=batch_size,
    shuffle=False
)

class_names = train_data.class_names
print(f'Class names: {class_names}')#21 classes

num_classes = len(class_names)

#-------model------

layers = tf.keras.layers
model = tf.keras.Sequential([
    layers.Rescaling(1./255),                                   #change of the numerical scale from 0-255 to 0-1
    layers.Conv2D(32, 3, activation= 'relu', padding = 'same'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation= 'relu', padding = 'same'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation= 'relu', padding = 'same'),
    layers.MaxPooling2D(),
    layers.GlobalAveragePooling2D(),                            #risk of overfitting with Flatten, this reduces the dimensions and keeps the important info with the average of each feature map
    layers.Dense(64, activation = 'relu'),
    layers.Dropout(0.3),                                        #30% of the neurons will be ignored during training to prevent overfitting
    layers.Dense(num_classes, activation = 'softmax')
])

model.compile(optimizer='adam',
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits= False),
              metrics=['accuracy'])

logdir = 'logs'
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=logdir,
    histogram_freq=1,
    write_images=logdir
 )
metrics_callback = MetricsCallback(
    val_data=val_data,
    class_names=class_names,
    output_dir="metrics"
)


model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[tensorboard_callback,
               metrics_callback]
)

model.save('cnn_model_1.h5')
model.summary()

"""
idéal 
train_acc ↑
val_acc ↑
train_loss ↓
val_loss ↓

overfitting
train_acc ↑↑
val_acc → ou ↓
train_loss ↓
val_loss ↑

underfitting
train_acc bas
val_acc bas


"""

