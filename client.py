from sklearn.utils import class_weight
import tensorflow_privacy as tfp
import tensorflow_privacy.privacy.optimizers.dp_optimizer as dp_optimizers
from tensorflow_privacy.v1 import DPGradientDescentGaussianOptimizer
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications import VGG16
import os
import flwr as fl
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_privacy as tfp
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import ResNet50


def load_data(train_dir, valid_dir, test_dir, class_imbalance=False, batch_size=4):
    train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
    rotation_range=8, 
    width_shift_range=0.01,  
    height_shift_range=0.01)  
    valid_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        classes=None,
        class_mode='categorical'
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        classes=None,
        class_mode='categorical'
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        classes=None,
        class_mode='categorical'
    )

    return train_generator, valid_generator, test_generator

epochs = 10
batch_size = 4

l2_norm_clip = 1.0
noise_multiplier = 0.001
learning_rate = 0.00001


data_folder = 'C:\\Users\\purvi\\Downloads\\Chest Data'
train_dir, valid_dir, test_dir = os.path.join(data_folder, 'train'), os.path.join(data_folder, 'valid'), os.path.join(data_folder, 'test')
train_generator, valid_generator, test_generator = load_data(train_dir, valid_dir, test_dir, class_imbalance=True)

base_model = ResNet50(
    input_shape=(224, 224, 3), weights="imagenet", include_top=False
)
for layer in base_model.layers[:-4]:  
    layer.trainable = False

flatten_in = Flatten()(base_model.output)
prediction = Dense(units=4, activation='softmax')(flatten_in)
prediction = Dropout(0.3)(prediction)  

model = Model(inputs=base_model.input, outputs=prediction)

optimizer = DPKerasAdamOptimizer(
    l2_norm_clip=1.0,
    noise_multiplier=0.005,
    num_microbatches=1,  
    learning_rate=0.00001
)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

class_weights = class_weight.compute_class_weight(
    class_weight='balanced', classes=np.unique(train_generator.classes), y=train_generator.classes
)
class_weights_dict = dict(enumerate(class_weights))


loss = tf.keras.losses.CategoricalCrossentropy(
    from_logits=False, reduction=tf.losses.Reduction.NONE)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
checkpoint = ModelCheckpoint('model_checkpoint.h5', save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
model.fit(train_generator,
          epochs=epochs,
          validation_data=valid_generator,
          batch_size=batch_size,
          class_weight=class_weights_dict,
          callbacks=[checkpoint, early_stopping])

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        early_stopping = EarlyStopping(monitor='val_loss', patience=1)

        # Train the model with evaluation on local data
        history = model.fit(train_generator,
                            epochs=30,
                            batch_size=4,
                            validation_data=valid_generator,
                            class_weight=class_weights_dict,
                            callbacks=[early_stopping])

        # Extract training and validation metrics
        train_loss = history.history['loss']
        train_accuracy = history.history['accuracy']
        val_loss = history.history['val_loss']
        val_accuracy = history.history['val_accuracy']

        # Return model updates, number of samples, and evaluation metrics
        return model.get_weights(), len(train_generator), {
            "train_loss": train_loss[-1],
            "train_accuracy": train_accuracy[-1],
            "val_loss": val_loss[-1],
            "val_accuracy": val_accuracy[-1]
        }
    

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(test_generator)
        return loss, len(test_generator), {"accuracy": accuracy}

fl.client.start_numpy_client(
    server_address="127.0.0.1:8080", client=FlowerClient()
)