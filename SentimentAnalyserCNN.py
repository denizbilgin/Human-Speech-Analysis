import time
import numpy as np
import os
import matplotlib.pyplot as plt
import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def plot_loss_accuracy(history, save_model):
    plt.figure(figsize=(10, 6))

    plt.title("Model Accuracy")
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    if save_model:
        plt.savefig("acc-epoch.png", dpi=300)
    plt.show()

    plt.title("Model Loss")
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    if save_model:
        plt.savefig("loss-epoch.png", dpi=300)
    plt.show()


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
            self.model.save(f"models/sentiment_analyser_CNN{logs.get('accuracy')*100:.2f}Acc.h5")
            print("The model is successfully saved.")


if __name__ == '__main__':

    BASE_DIR = "data/images/emotions/"
    EMOTIONS = os.listdir(BASE_DIR)
    NUM_CLASSES = len(EMOTIONS)
    IMAGE_SIZE = (96, 96)
    BATCH_SIZE = 16
    EPOCHS = 300
    SAVE_MODEL = True
    SHOW_TRAINING_STATISTICS = True
    USE_SAVED_MODEL = False
    COLOR_MODE = "rgb"

    print(f"Contents of train dir: {EMOTIONS}")

    datagen = ImageDataGenerator(
        rescale=1.0/255,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        BASE_DIR,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=IMAGE_SIZE,
        shuffle=True,
        subset="training",
        color_mode=COLOR_MODE
    )
    validation_generator = datagen.flow_from_directory(
        BASE_DIR,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=IMAGE_SIZE,
        shuffle=True,
        subset="validation",
        color_mode=COLOR_MODE
    )

    sample_batch = next(train_generator)
    print(f"Shape of sample batch is : {sample_batch[0].shape}")

    first_img = sample_batch[0][0]
    img_label = EMOTIONS[np.argmax(sample_batch[1][0])]
    plt.imshow(first_img)
    plt.title("First image of training set (" + img_label + ")\n" + str(first_img.shape))
    plt.axis('off')
    plt.show()

    if USE_SAVED_MODEL:
        model = keras.models.load_model("models/sentiment_analyser_CNN63.52Acc.h5")
    else:
        model = keras.models.Sequential([
            keras.layers.Input(shape=(96, 96, 1 if COLOR_MODE == "grayscale" else 3)),
            keras.layers.Conv2D(16, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(256, (3, 3), activation="relu"),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dense(512, activation="elu", kernel_initializer="lecun_normal",
                               kernel_regularizer=keras.regularizers.l2(0.016)),
            keras.layers.Dropout(0.3),
            keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
            keras.layers.Dense(256, activation="elu", kernel_initializer="lecun_normal",
                               kernel_regularizer=keras.regularizers.l2(0.016)),
            keras.layers.Dense(NUM_CLASSES, activation="softmax")
        ])

    model.compile(optimizer=keras.optimizers.Adamax(learning_rate=3e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print(model.summary())

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator,
        callbacks=[
            SentimentAnalyserCallback(SAVE_MODEL)
        ]
    )

    if SHOW_TRAINING_STATISTICS:
        plot_loss_accuracy(history, SAVE_MODEL)

    # loss, acc = model.evaluate(test_generator)
    # print(acc)