from PIL import Image
from keras import layers
from sklearn import metrics
from time import gmtime, strftime
import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

def arguments():
    current_date = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-resize', '-s', default='64', required=False,
                        type=int,
                        help="Specifies the size to which images will be \
                        resized to. Remember: the bigger the images/resize, \
                        the more memory will be required and the long it'll \
                        take to chew through everything.\nDefault: 64x64")
    parser.add_argument('--optimizer', '-o', default='adam', required=False,
                        type=str,
                        help="Specifies the optimized to use in the model. \
                        \nDefault: adam")
    parser.add_argument('--epochs', '-e', default=50, required=False,
                        type=int,
                        help="Specifies the number of epochs to iterate through \
                        during the model's training.\nDefault: 50")
    parser.add_argument('--saved-model', '-f', default=f"{current_date}.keras",
                        required=False, type=str,
                        help="Specify the saved model's path and filename. \
                             \nDefault: %Y-%m-%dT%H:%M:%SZ.keras")
    parser.add_argument('--path-test', '-pt', default=None, required=True,
                        type=str,
                        help="Set the file containing the 'path, category' of \
                        images to be tested against.\nDefault: None")
    parser.add_argument('--path-train', '-ptr', default=None, required=True,
                        type=str,
                        help="Set the file containing the 'path, category' of \
                        images to be trained against\nDefault: None")
    parser.add_argument('--backend', '-b', default=None, required=True,
                        type=str,
                        help="Set matplotlib's pyplot backend.\nDefault: \
                        GTK3Agg")
    return parser.parse_args()

def load_pltconfigs():
    return plt.switch_backend(arguments().backend)

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
def populate_array(csv_path, image_array, label_array):
        rescaling_RGB = layers.Rescaling(1./255)
        with open(csv_path) as csvfile:
            conjunto = csv.reader(csvfile)
            for path, label in conjunto:
                print(f"Path: {path}\n Label: {label}")
                img = Image.open(path)
                img = img.resize((arguments().image_resize,arguments().image_resize))
                img_as_array = np.asarray(img)
                image_array.append(img_as_array)
                label_array.append(convert_label(label))
        image_array = np.array(rescaling_RGB(image_array))

def generate_arrays():
    images_test = []
    labels_test = []
    images_train = []
    labels_train = []
    populate_array(arguments().path_test, images_test, labels_test)
    populate_array(arguments().path_train, images_train, labels_train)
    labels_train = np.array(labels_train, dtype='float32')
    images_train = np.array(images_train, dtype='uint8')
    labels_test = np.array(labels_train, dtype='float32')
    images_test = np.array(images_train, dtype='uint8')
    return images_test, labels_test, images_train, labels_train

def show_sample():
    _, _, images_train, _ = generate_arrays()
    load_pltconfigs()
    plt.imshow(images_train[1].astype("uint8"))
    plt.title("lmao")
    plt.show()

def train_model():
    _, _, images_train, labels_train = generate_arrays()
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(arguments().image_resize,
                                             arguments().image_resize, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer=arguments().optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(images_train, labels_train, epochs=arguments().epochs)

def test_model():
    images_test, labels_test, _, _ = generate_arrays()
    test_loss, test_acc = model.evaluate(images_test,  labels_test, verbose=2)
    print('\nTest accuracy:', test_acc)

def save_model():
    model.save(arguments().saved_model)

def predict_model():
    images_test, labels_test, _, _ = generate_arrays()
    predictions = model.predict(images_test)
    metrics.accuracy_score(np.argmax(predictions, axis=1), labels_test)
    print(f'\nPredictions: {predictions}')

if __name__ == '__main__':
    show_sample()
    train_model()
    save_model()
    test_model()
    predict_model()
