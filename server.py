import flwr as fl
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2  
from tensorflow.keras.layers import Dense, Flatten
import tensorflow_privacy as tfp
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasAdamOptimizer
from flwr.server.client_manager import SimpleClientManager
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import ResNet50

epsilon = 2.0
delta = 1e-5
noise_multiplier = 0.001  
num_microbatches = 1  
l2_norm_clip = 1.0  

def flower_server_model():
    base_model = ResNet50(
        input_shape=(224, 224, 3),
        weights='imagenet',
        include_top=False
    )
    for layer in base_model.layers:
        layer.trainable = False

    flatten_in = Flatten()(base_model.output)
    prediction = Dense(units=4, activation='softmax')(flatten_in)
    prediction = Dropout(0.3)(prediction)  


    server_model = Model(inputs=base_model.input, outputs=prediction)
    optimizer = DPKerasAdamOptimizer(
        l2_norm_clip=1.0,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=0.00001,
        clipnorm=1.0
    )

    server_model.compile(optimizer=optimizer, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])

    return server_model

class FlowerServerWithDP(fl.server.Server):
    def __init__(self, model, noise_multiplier, num_microbatches, epsilon, delta, client_manager):
        super().__init__(client_manager=client_manager)
        self.model = model
        self.noise_multiplier = noise_multiplier
        self.num_microbatches = num_microbatches
        self.epsilon = epsilon
        self.delta = delta
        

client_manager = SimpleClientManager()
server_model = flower_server_model()

server = FlowerServerWithDP(
    model=server_model,
    noise_multiplier=noise_multiplier,
    num_microbatches=num_microbatches,
    epsilon=epsilon,
    delta=delta,
    client_manager=client_manager
)

# Flower server with FedAvg strategy
fl.server.start_server(
    server_address='0.0.0.0:8080',
    config=fl.server.ServerConfig(num_rounds=15),  
    strategy=fl.server.strategy.FedAvg(),
    server=server,
    client_manager=client_manager
)
