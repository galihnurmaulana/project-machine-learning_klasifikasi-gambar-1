import tensorflow as tf
print(tf.__version__)

#Unduh Dataset
!wget https://github.com/dicodingacademy/assets/releases/download/release/rockpaperscissors.zip

#Ekstrak Dataset
!unzip rockpaperscissors.zip

#Membagi dataset menjadi train set dan validation set dengan mengimplementasikan augmentasi gambar dan menggunakan image data generator
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    validation_split=0.4,
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = datagen.flow_from_directory(
    'rockpaperscissors/rps-cv-images',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    'rockpaperscissors/rps-cv-images',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

#Bangun model sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(3))
model.add(Activation('softmax'))

#Kompilasi dan Latih Model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(train_generator,
                    steps_per_epoch=25,
                    epochs=20,
                    validation_data=validation_generator,
                    validation_steps=5)

#Evaluasi Model
test_loss, test_acc = model.evaluate(validation_generator)
print(f'Validation accuracy: {test_acc * 100:.2f}%')

from google.colab import files
from IPython.display import Image
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline

# Mengunggah gambar
uploaded = files.upload()

# Menampilkan gambar yang diunggah
for filename in uploaded.keys():
    img_path = filename
    img = Image(filename=img_path)
    display(img)

# Memprediksi gambar yang diunggah
img = image.load_img(img_path, target_size=(150, 150))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
prediction = model.predict(img)
class_index = np.argmax(prediction)
classes = ['rock', 'paper', 'scissors']

print(f'Predicted class: {classes[class_index]}')
