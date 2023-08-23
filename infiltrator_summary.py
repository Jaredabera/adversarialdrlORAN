import tensorflow as tf

autoencoder = tf.keras.models.load_model('D:\DRL based Resource Allocation in\drl_ra_adv_attack\ResourceAllocatorSaboteur\ml_models\encoder.h5')

autoencoder.summary()
