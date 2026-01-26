import logging
import sys
import keras

import tensorflow as tf
from pathlib import Path
from tqdm.auto import tqdm

root_path = Path(__file__).resolve().parent.parent
filename = Path(__file__).stem
sys.path.append(str(root_path))

from set_logging import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(filename)

# Model init
base = keras.applications.ResNet50(
    include_top = False,
    weights = 'imagenet',
    pooling = 'avg' # Switch AveragePooling2D instead Flatten
)                   # Output shape = (batch_size, 2048)

# Construct and initialize custom head for two classes
base_inputs = keras.layers.Input((2048, ))
dropout = keras.layers.Dropout(0.3)(base_inputs)
regresssor = keras.layers.Dense(
    units = 1024,
    activation = 'relu'
)(dropout)
model_outputs = keras.layers.Dense(
    units = 2
)(regresssor)
head = keras.Model(
    inputs = base_inputs,
    outputs = model_outputs
) # Input -> Dropout -> Regressor (Dense + ReLU) -> Output


# Union construction
class ResNet50_with_classifier(keras.Model):
    def __init__(self, conv_base, classifier):
        self.conv_base = conv_base
        self.head = classifier
        logger.debug('Full ResNet initialized') 


    def call(self, inputs, training = True):
        conv_outs = self.conv_base(inputs, training = training)
        output = self.classifier(conv_outs, training = training)
        return output # Raw logits. Shape = (batch_size, 2)
    
