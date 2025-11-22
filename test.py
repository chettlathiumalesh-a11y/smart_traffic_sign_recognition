import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("traffic_sign_model.h5")

img_path = "test_image.jpg"   # change image name here
img = cv2.imread(img_path)
img_resized = cv2.resize(img, (64, 64))
img_resized = img_resized / 255.0
img_resized = np.expand_dims(img_resized, axis=0)

prediction = model.predict(img_resized)
class_id = np.argmax(prediction)

print("Predicted Traffic Sign Class:", class_id)
