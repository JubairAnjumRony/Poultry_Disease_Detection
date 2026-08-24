import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts", "training", "model.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image / 255.0, axis=0)
        probabilities = model.predict(test_image, verbose=0)[0]
        result = int(np.argmax(probabilities))
        confidence = float(probabilities[result])

        if result == 1:
            prediction = 'Healthy'
            return [{"image": prediction, "prediction": prediction, "confidence": confidence}]
        else:
            prediction = 'Coccidiosis'
            return [{"image": prediction, "prediction": prediction, "confidence": confidence}]