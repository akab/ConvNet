import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


class TFHandler:
    def __init__(self):
        self.session = tf.Session()
        self.list_devices(self.session)

    def configure(self):
        print("TensorFlow version:", tf.__version__)
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.95
        config.gpu_options.visible_device_list = "0"
        # Configure session
        self.session = tf.Session(config=config)
        set_session(self.session)
        print("TensorFlow configuration:\n", config.gpu_options)

    @staticmethod
    def list_devices(sess):
        if not sess:
            return

        devices = sess.list_devices()
        for d in devices:
            print(d.name)

    def close(self):
        self.session.close()
        self.session.__del__()
