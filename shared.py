import tensorflow as tf
from enum import Enum

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

class TrainingType(Enum):
    SCRATCH = 1
    IMGNET = 2
    RESUME = 3