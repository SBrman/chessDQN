import numpy as np
import tensorflow as tf

from enum import Enum, auto
from moves import ALL_LEGAL_MOVES

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

from send_slack import send_msg
from pprint import pformat


class QType(Enum):
    DQN = auto()
    TABULAR = auto()


class DeepQNetwork:
    """Using a Convolutional Neural Network Architecture."""
    def __init__(self, epochs: int = 5, batch_size: int = 200, saved_model: str = None):
        self.name = 'cnn_chessRL'

        self.optimizer = keras.optimizers.Adam(learning_rate=0.00005)#, clipnorm=1.0)
        self.loss_function = keras.losses.Huber()
        self.epochs = epochs
        self.batch_size = batch_size

        self.training_count = 0

        if not saved_model:
            self.model = self.create_model(batch_size=self.batch_size)
        else:
            self.model = keras.models.load_model(saved_model)
        
        self.predictor_model = self.create_model(batch_size=1)
        
        self.predictor_model.set_weights(self.model.get_weights())
        
        self.frozen_func = None
        self.save_as_function(self.predictor_model)
        self.load_frozen_graph()
    
    def __repr__(self):
        return str(self.model.summary())
    
    def create_model(self, batch_size: int):
        input_layer = layers.Input(shape=(8, 8, 1), batch_size=batch_size)
        
        hidden_layer_1 = layers.Conv2D(32, (3, 3), strides=1, activation='relu')(input_layer)
        hidden_layer_2 = layers.Conv2D(64, (3, 3), strides=1, activation='relu')(hidden_layer_1)
        hidden_layer_3 = layers.Conv2D(64, (2, 2), strides=1, activation='relu')(hidden_layer_2)
        
        hidden_layer_4 = layers.Flatten()(hidden_layer_3)
        hidden_layer_5 = layers.Dense(512, activation='relu')(hidden_layer_4)
        
        output_layer = layers.Dense(len(ALL_LEGAL_MOVES), activation='linear')(hidden_layer_5)
        
        model = keras.Model(inputs=input_layer, outputs=output_layer)

        if batch_size != 1:
            model.compile(optimizer=self.optimizer, loss=self.loss_function, metrics=['accuracy'])

        return model
    
    def train(self, features_and_labels, save_path):
        """Features is board state and the labels is the action q values."""

        features, labels = list(zip(*features_and_labels))
        features = np.array(features)
        labels = np.array(labels)

        res = self.model.fit(features, labels, epochs=self.epochs, batch_size=self.batch_size, shuffle=True)
        send_msg(str(pformat(res.history, indent=4)))
        
        self.model.save(save_path)
        self.predictor_model.set_weights(self.model.get_weights())
        self.save_as_function(self.predictor_model)
        self.load_frozen_graph()

        self.training_count += 1
    
    @staticmethod 
    def save_as_function(model):
        """Predicting directly from keras model is super slow. This method turns keras model into a frozen function
        which is later used to make faster predictions."""
        
        # Convert Keras model to ConcreteFunction
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

        # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()

        # Save frozen graph from frozen ConcreteFunction to hard drive
        tf.io.write_graph(graph_or_graph_def=frozen_func.graph, logdir="./frozen_models", name="frozen_graph.pb", as_text=False) 
    
    @staticmethod 
    def wrap_frozen_graph(graph_def, inputs, outputs):
        """Returns the frozen function"""
        _imports_graph_def = lambda: tf.compat.v1.import_graph_def(graph_def, name="")
        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph

        return wrapped_import.prune(tf.nest.map_structure(import_graph.as_graph_element, inputs), 
                                    tf.nest.map_structure(import_graph.as_graph_element, outputs))
 
    def load_frozen_graph(self):
        """Loads frozen graph, creates and then returns the frozen function using the loaded frozen graph."""
        with tf.io.gfile.GFile("./frozen_models/frozen_graph.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        self.frozen_func = self.wrap_frozen_graph(graph_def=graph_def, inputs=["x:0"], outputs=["Identity:0"])
        
    def get_Q(self, state):
        """Make predictions."""
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        state = tf.reshape(state, self.predictor_model.inputs[0].shape)
        
        return self.frozen_func(state)[0][0]
