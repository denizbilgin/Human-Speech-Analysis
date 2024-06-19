import time
import numpy as np
import os
import matplotlib.pyplot as plt
import keras
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def plot_loss_accuracy(history, save_model):
    plt.figure(figsize=(12, 5))
    print(history.history["f1_score"])
    # Model Accuracy
    plt.subplot(1, 2, 1)
    plt.title("Model F1 Score")
    plt.plot(history.history["f1_score"])
    plt.plot(history.history["val_f1_score"])
    plt.ylabel("F1 Score")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")

    # Model Loss
    plt.subplot(1, 2, 2)
    plt.title("Model Loss")
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc="upper left")

    # Save the figure if required
    if save_model:
        final_val_f1 = str(tf.reduce_mean(history.history["val_f1_score"][-1]).numpy())
        final_val_f1 = final_val_f1[2:4] + "-" + final_val_f1[4:6]
        file_name = f"statistics_of_{final_val_f1}F1_model.png"
        plt.savefig("./model-statistics/" + file_name, dpi=300)

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, generator):
    y_val_pred = model.predict(generator, steps=generator.samples // generator.batch_size + 1)
    y_val_pred_classes = np.argmax(y_val_pred, axis=1)
    y_val_true = generator.classes
    cm = tf.math.confusion_matrix(y_val_true, y_val_pred_classes)
    class_names = list(generator.class_indices.keys())
    plt.figure(figsize=(14, 9))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

class SentimentAnalyserCallback(keras.callbacks.Callback):
    def __init__(self, save_model):
        super().__init__()
        self.save_model = save_model
        self.start_time = None

    def on_train_begin(self, logs=None):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        mean_f1 = tf.reduce_mean(logs.get("val_f1_score")).numpy()
        if logs is not None and mean_f1 > 0.85:
            print(f"\nReached 85% F1 Score, training stopped..")
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        training_duration = time.time() - self.start_time
        hours = training_duration // 3600
        minutes = (training_duration - (hours * 3600)) // 60
        seconds = training_duration - ((hours * 3600) + (minutes * 60))

        message = f"Training elapsed time was {str(hours)} hours, {minutes:4.1f} minutes, {seconds:4.2f} seconds."
        print(message)

        last_val_f1 = str(tf.reduce_mean(logs.get("val_f1_score")).numpy())
        last_val_f1 = last_val_f1[2:4] + "-" + last_val_f1[4:6]
        if logs is not None and self.save_model:
            self.model.save(f"models/sentiment_analyser_CNN{last_val_f1}F1.h5")
            print("The model is successfully saved.")

if __name__ == '__main__':
    # General constants
    BASE_DIR = "data/images/emotions/"
    EMOTIONS = os.listdir(BASE_DIR)
    NUM_CLASSES = len(EMOTIONS)
    IMAGE_SIZE = (96, 96)
    BATCH_SIZE = 32
    EPOCHS = 50
    SAVE_MODEL = True
    USE_SAVED_MODEL = False
    SHOW_TRAINING_STATISTICS = not USE_SAVED_MODEL
    COLOR_MODE = "rgb"

    print(f"Contents of train dir: {EMOTIONS}")
    print("-------------------------------")

    # Creating train and validation datasets
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
    print("----------------------------------------")

    # Checking whether the data is balanced or not
    train_classes = train_generator.classes
    unique_classes, class_counts = np.unique(train_classes, return_counts=True)
    for cls, count in zip(unique_classes, class_counts):
        print(f"There are {count} images for {EMOTIONS[cls]}.")
    print("\nAs you can see the data is unbalanced.")
    print("----------------------------------------")

    sample_batch = next(train_generator)
    print(f"Shape of sample batch is : {sample_batch[0].shape}")
    print("----------------------------------------")

    # Plotting the first image of the training generator to getting insight
    first_img = sample_batch[0][0]
    img_label = EMOTIONS[np.argmax(sample_batch[1][0])]
    plt.imshow(first_img)
    plt.title("First image of training set (" + img_label + ")\n" + str(first_img.shape))
    plt.axis('off')
    plt.show()

    if USE_SAVED_MODEL:
        model = keras.models.load_model("models/sentiment_analyser_CNN.h5")
        print(model.summary())
    else:
        # Creating deep convolutional neural network architecture
        model = keras.models.Sequential([
            keras.layers.Input(shape=(96, 96, 1 if COLOR_MODE == "grayscale" else 3)),
            keras.layers.Conv2D(32, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(512, (3, 3), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation="relu", kernel_regularizer=keras.regularizers.l1_l2()),
            keras.layers.Dropout(0.4),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(64, activation="relu", kernel_regularizer=keras.regularizers.l1_l2()),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(NUM_CLASSES, activation="softmax")
        ])

        print(model.summary())

        # Compiling and setting the model
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
                      loss='categorical_crossentropy',
                      metrics=['f1_score', "accuracy"])

        history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=validation_generator,
            callbacks=[
                SentimentAnalyserCallback(SAVE_MODEL)
            ]
        )

        if SHOW_TRAINING_STATISTICS:
            # Showing statistics about training and the model
            plot_loss_accuracy(history, SAVE_MODEL)

    # Testing the model by dummy generator
    test_generator = datagen.flow_from_directory(
        BASE_DIR,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        target_size=IMAGE_SIZE,
        shuffle=False,
        subset="validation",
        color_mode=COLOR_MODE
    )
    plot_confusion_matrix(model, test_generator)