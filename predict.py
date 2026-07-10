import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model.h5")

def predict_digit(path):

    img = cv2.imread(path,0)

    img = cv2.resize(img,(28,28))

    img = cv2.bitwise_not(img)

    img = img/255.0

    img = img.reshape(1,28,28)

    prediction = model.predict(img)

    digit = np.argmax(prediction)

    confidence = np.max(prediction)

    return digit,confidence