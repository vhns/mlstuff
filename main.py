from PIL import Image
from keras import layers
from sklearn import metrics
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

path_test='./filelist_test.txt'
path_train='./filelist_train.txt'
plt.switch_backend('GTK3Agg')

images_test = []
labels_test = []
images_train = []
labels_train = []

if not (path_train, path_test):
    print("Missing files")
    exit()

def convert_label(label):
    if label:
        if label == 'Empty':
            return 0
        if label == 'Occupied':
            return 1
        else:
            return None
    else:
        print("missing label")

# Change this to rely exclusively on Tensorflow's
# tf.keras.utils.load_img and  tf.keras.utils.img_to_array
# (Should be better performant and reliable(?)).
def generate_arrays(csv_path, image_array, label_array):
        rescaling_RGB = layers.Rescaling(1./255)
        with open(csv_path) as csvfile:
            conjunto = csv.reader(csvfile)
            for path, label in conjunto:
                print(f"Path: {path}\n Label: {label}")
                img = Image.open(path)
                img = img.resize((64,64))
                img_as_array = np.asarray(img)
                image_array.append(img_as_array)
                label_array.append(convert_label(label))
        image_array = np.array(rescaling_RGB(image_array))


generate_arrays(path_test, images_test, labels_test)
generate_arrays(path_train, images_train, labels_train)

#with open(path_train) as csvfile:
#    conjunto = csv.reader(csvfile)
#    for path, label in conjunto:
#        print(f"Path: {path}\n Label: {label}")
#        img = Image.open(path)
#        img = img.resize((64,64))
#        img_as_array = np.asarray(img)
#        images_train.append(img_as_array)
#        labels_train.append(convert_label(label))
#
#with open(path_test) as csvfile:
#    conjunto = csv.reader(csvfile)
#    for path, label in conjunto:
#        print(f"Path: {path}\n Label: {label}")
#        img = Image.open(path)
#        img = img.resize((64,64))
#        img_as_array = np.asarray(img)
#        images_test.append(img_as_array)
#        labels_test.append(convert_label(label))
#
labels_train = np.array(labels_train, dtype='float32')
images_train = np.array(images_train, dtype='uint8')
labels_test = np.array(labels_train, dtype='float32')
images_test = np.array(images_train, dtype='uint8')

print(images_train.shape)
print(labels_train.shape)

plt.imshow(images_train[1].astype("uint8"))
plt.title("lmao")
plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(64, 64, 3)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(images_train, labels_train, epochs=50)

test_loss, test_acc = model.evaluate(images_test,  labels_test, verbose=2)

print('\nTest accuracy:', test_acc)


# As "activation=max" is being specificied in "model = ",
# there's no real need to create a prediction model with softmax specified.
#probability_model = tf.keras.Sequential([model,
#                                         tf.keras.layers.Softmax()])
model.save('asjdlkasjld.hdf5')
predictions = model.predict(images_test)



metrics.accuracy_score(np.argmax(predictions, axis=1), labels_test)


print(f'\nPredictions: {predictions}')

with open('test_save_model.img', 'wb') as f:
    f.write()
