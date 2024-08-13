import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix
from tensorflow.keras.applications.densenet import DenseNet121
import numpy as np
import matplotlib.pyplot as plt

def address_class_imbalance(train_generator):
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    train_generator.class_weight = dict(enumerate(class_weights))


if __name__ == "__main__":
    data_folder = 'C:\\Users\\purvi\\Downloads\\Chest Data'
    train_dir = os.path.join(data_folder, 'train')
    valid_dir = os.path.join(data_folder, 'valid')
    test_dir = os.path.join(data_folder, 'test')

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,  # Normalization
        shear_range=0.2,
        zoom_range=0.4,
        horizontal_flip=True,
        rotation_range=30,
        width_shift_range=0.4,
        height_shift_range=0.4,
        preprocessing_function=tf.keras.applications.densenet.preprocess_input  # For DenseNet
    )
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                    preprocessing_function=tf.keras.applications.densenet.preprocess_input)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,
                                                                   preprocessing_function=tf.keras.applications.densenet.preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),  # Resizing
        batch_size=16,
        class_mode='categorical',
        shuffle=True
    )
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )
    address_class_imbalance(train_generator)
    base_model = tf.keras.applications.densenet.DenseNet121(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=False
    )
    for layer in base_model.layers:
        layer.trainable = False

    # Add top layers
    flatten_in = Flatten()(base_model.output)
    prediction = Dense(units=4, activation='softmax')(flatten_in)

    # Full model compilation
    full_model = Model(inputs=base_model.input, outputs=prediction)
    full_model.compile(
       optimizer=Adam(learning_rate=0.0001),
       loss='categorical_crossentropy',
       metrics=['accuracy']
    )

    # Learning rate scheduling and early stopping
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3),
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]

    # Train and evaluate model
    history = full_model.fit(
        train_generator,
        epochs=20,
        validation_data=valid_generator,
        callbacks=callbacks
    )

    # Evaluate on the test set
    test_loss, test_acc = full_model.evaluate(test_generator)
    print('Test accuracy:', test_acc)

    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

    # Predictions on the test set
    y_true = test_generator.classes
    y_pred_prob = full_model.predict(test_generator)
    y_pred = np.argmax(y_pred_prob, axis=1)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print('Confusion Matrix:')
    print(cm)
