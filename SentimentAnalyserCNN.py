import numpy as np
import tensorflow as tf
import os
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

if __name__ == '__main__':
    base_dir = "data/images/emotions"

    print(f"Contents of train dir: {os.listdir(base_dir + '/train')}")

    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")

    train_datagen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0/255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        batch_size=32,
        class_mode="categorical",
        target_size=(48, 48)
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        batch_size=32,
        class_mode="categorical",
        target_size=(48, 48)
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(48, 48, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(7, activation='softmax')
    ])

    model.compile(optimizer="adam",
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        train_generator,
        epochs=15
    )

    loss, acc = model.evaluate(test_generator)
    print(acc)

    # batch = next(test_generator)
    # first_img = batch[0][0]
    #
    # plt.imshow(first_img)
    # plt.axis('off')  # Eksenleri kapatın
    # plt.show()