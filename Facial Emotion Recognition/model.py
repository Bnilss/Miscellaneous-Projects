from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.compat.v1.Session(config=config)

class FacialExpressionModel(object):
    EMOTION = ("Angry",'Disgust','Fear','Happy','Neutral','Sad','Suprised')
    def __init__(self, model, weights):
        with open(model) as f:
            self.model = model_from_json(f.read())

        self.model.load_weights(weights)
        self.model._make_predict_function()

    def predict_emotion(self, image):
        self.preds = self.model.predict(image)
        return self.EMOTION[np.argmax(self.preds)]
