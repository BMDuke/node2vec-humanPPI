import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package='Custom', name=None)
def walk_serializer(x): 
    return x