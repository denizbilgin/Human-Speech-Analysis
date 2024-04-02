import time
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import keras
from PIL import Image
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
    EMOTIONS = ["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"]
    NUM_CLASSES = len(EMOTIONS)
    BASE_DIR = "data/images/emotions"
    TRAIN_DIR = os.path.join(BASE_DIR, "train")
    TEST_DIR = os.path.join(BASE_DIR, "test")
    IMAGE_SIZE = (48, 48)
    BATCH_SIZE = 32
    EPOCHS = 50
    SAVE_MODEL = True
    SHOW_STATISTICS = True
    USE_SAVED_MODEL = False


    print(f"Contents of train dir: {os.listdir(BASE_DIR + '/train')}")

    datagen = ImageDataGenerator(
        rescale=1.0/255,
        horizontal_flip=True,
        rotation_range=20,
        shear_range=0.2,
        zoom_range=0.3,
        fill_mode="nearest",
        validation_split=0.15
    )
    test_datagen = ImageDataGenerator(
        rescale=1.0/255
    )

    train_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=IMAGE_SIZE,
        shuffle=True,
        subset="training"
    )
    validation_generator = datagen.flow_from_directory(
        TRAIN_DIR,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=IMAGE_SIZE,
        shuffle=True,
        subset="validation"
    )
    test_generator = test_datagen.flow_from_directory(
        TEST_DIR,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=IMAGE_SIZE,
        shuffle=True
    )

    sample_batch = next(train_generator)
    print(f"Shape of sample batch is : {sample_batch[0].shape}")

    first_img = sample_batch[0][0]
    img_label = EMOTIONS[np.argmax(sample_batch[1][0])]
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
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    print(model.summary())

    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[SentimentAnalyserCallback(SAVE_MODEL)]
    )

    loss, acc = model.evaluate(test_generator)
    print(acc)

