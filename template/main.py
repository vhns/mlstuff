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
import pdb

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
    parser.add_argument('--shuffle-size', '-z', default=100, required=False,
                        type=int,
                        help="Specifies the default shuffle size.\nDefault: 100")
    parser.add_argument('--batch-size', '-k', default=32, required=False,
                        type=int,
                        help="Specifies the default batch size.\nDefault: 32")
    parser.add_argument('--saved-model', '-f', default=f"./{current_date}.keras",
                        required=False, type=str,
                        help="Specify the saved model's path and filename. \
                             \nDefault: %Y-%m-%dT%H:%M:%SZ.keras")
    parser.add_argument('--path-test', '-pt', default=None, required=True,
                        type=str,
                        help="Set the file containing the 'path, category' of \
                        images to be tested against.\nDefault: None")
    parser.add_argument('--path-checkpoint', '-pc', default=f"/tmp/{current_date}", required=False,
                        type=str,
                        help="Set the path, in which the datasets checkpoints will be.\nDefault: /tmp/$current_date")
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
        if label == None:
            return 'Empty'
        if label == 1:
            return 'Occupied'
        else:
            return None
    else:
        return 'Empty'

# Change this to rely exclusively on Tensorflow's
# tf.keras.utils.load_img and  tf.keras.utils.img_to_array
# (Should be better performant and reliable(?)).
def populate_array(csv_path, image_array, label_array):
        rescaling_RGB = layers.Rescaling(1./255)
        with open(csv_path) as csvfile:
            conjunto = csv.reader(csvfile)
            for path, label in conjunto:
                #print(f"Path: {path}\n Label: {label}")
                img = Image.open(path)
                img = img.resize((arguments().image_resize,arguments().image_resize))
                img_as_array = np.asarray(img)
                image_array.append(img_as_array)
                label_array.append(convert_label(label))
        return np.array(rescaling_RGB(image_array))

def generate_array(path):
    images = []
    labels = []
    populate_array(path, images, labels)
    labels = np.array(labels, dtype='float32')
    images = np.array(images, dtype='uint8')
    return images, labels

def generate_dataset(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    #dataset = dataset.shuffle(arguments().shuffle_size).batch(arguments().batch_size)
    print(f"DATASET SIZE: {len(dataset)}")
    dataset = dataset.shuffle(len(dataset)).batch(arguments().batch_size)
    return dataset

def generate_dataset_test(images, labels):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    #dataset = dataset.shuffle(arguments().shuffle_size).batch(arguments().batch_size)
    print(f"DATASET SIZE: {len(dataset)}")
    dataset = dataset.shuffle(len(dataset))
    return dataset

def show_sample(dataset, images_batch, images_label):
    images_batch, images_label = next(iter(dataset))
    plt.figure(figsize=(10, 10))
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images_batch[i].numpy().astype("uint8"))
        label = images_label[i]
        label = convert_label(label.numpy())
        plt.title(label)
        plt.axis("off")
    plt.show()

def train_model(dataset):
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath = arguments().path_checkpoint,
        #        Disabled for now as we don't have a parser flag for it
        #        and I arbitrarily want to save the entire model for
        #        later inspection.
        #        save_weights_only = true
        
        #        Disabled per the same logic as above
        #        save_best_only=True
        monitor="val_accuracy",
        # Default value, but we'll specify it either way 
        save_freq="epoch")
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(arguments().image_resize,
                                             arguments().image_resize, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer=arguments().optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(dataset, epochs=arguments().epochs)
    return model

def test_model(model, test_ds):
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print('\nTest accuracy:', test_acc)

def save_model(model):
    model.save(arguments().saved_model)

def predict_model(model, test_ds):
    predictions = model.predict(test_ds)
    pdb.set_trace()
    metrics.accuracy_score(np.argmax(predictions, axis=1), np.argmax(test_ds, axis=1))
    print(f'\nPredictions: {predictions}')

if __name__ == '__main__':
    test_images, test_labels = generate_array(arguments().path_test)
    train_images, train_labels = generate_array(arguments().path_train)
    test_ds = generate_dataset_test(test_images, test_labels)
    train_ds = generate_dataset(train_images, train_labels)
    load_pltconfigs()
    plt.switch_backend(arguments().backend)
    show_sample(train_ds, train_images, train_labels)
    model = train_model(train_ds)
    save_model(model)
    test_model(model, test_ds)
    predict_model(model, test_ds)
