import tensorflow as tf

autoencoder = tf.keras.models.load_model('./ml_models/encoder.h5')

autoencoder.summary()
