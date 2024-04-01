import time

import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class SentimentAnalyserCallback(keras.callbacks.Callback):
    def __init__(self, save_model):
        super().__init__()
        self.save_model = save_model
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None and logs.get("accuracy") > 0.99:
            print(f"\nReached 99% accuracy, training stopped..")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        training_duration = time.time() - self.start_time
        hours = training_duration // 3600
        minutes = (training_duration - (hours * 3600)) // 60
        seconds = training_duration - ((hours * 3600) + (minutes * 60))

        message = f"Training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds."
        print(message)

        if logs is not None and self.save_model:
            self.model.save(f"models/sentiment_analyser_CNN{logs.get('accuracy')}Acc.h5")
            print("The model is successfully saved.")


if __name__ == '__main__':
    emotions = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
    base_dir = "data/images/emotions"

    print(f"Contents of train dir: {os.listdir(base_dir + '/train')}")

    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    train_datagen = ImageDataGenerator(
        rescale=1.0/255,
        horizontal_flip=True,
        rotation_range=0.4,
        shear_range=0.2,
        zoom_range=0.1,
        fill_mode="nearest"
    )
    test_datagen = ImageDataGenerator(
        rescale=1.0/255
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=32,
        class_mode="categorical",
        target_size=(48, 48),
        shuffle=True
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        batch_size=32,
        class_mode="categorical",
        target_size=(48, 48),
        shuffle=True
    )

    sample_batch = next(train_generator)
    print(f"Shape of sample batch is : {sample_batch[0].shape}")

    first_img = sample_batch[0][0]
    img_label = emotions[np.where(sample_batch[1][0] == 1)[0]]
    plt.imshow(first_img)
    plt.title("First image of training set (" + img_label + ")")
    plt.axis('off')
    plt.show()

    model = keras.models.Sequential([
        keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(48, 48, 3)),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(128, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(7, activation='softmax')
    ])

    print(model.summary())

    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=70,
        callbacks=[SentimentAnalyserCallback(True)]
    )

    loss, acc = model.evaluate(test_generator)
    print(acc)

