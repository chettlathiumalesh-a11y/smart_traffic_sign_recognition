import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("traffic_sign_model.h5")

# Start camera (0 = default webcam)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    # Preprocess frame for prediction
    img = cv2.resize(frame, (64, 64))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    prediction = model.predict(img)
    class_id = np.argmax(prediction)

    # Display prediction on screen
    cv2.putText(frame, f"Class: {class_id}", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Traffic Sign Detection", frame)

    # Press "q" key to exit camera window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
