import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('models/sign_language_translator_model.h5')

# Define your class labels based on your dataset
class_labels = ['Hello', 'Thank You', 'Goodbye']  # Replace with your actual labels

# Function to preprocess the frame (customize as needed)
def preprocess_frame(frame):
    # Resize the frame to the input size expected by the model
    frame = cv2.resize(frame, (64, 64))  # Adjust size based on your model
    frame = frame.astype('float32') / 255.0  # Normalize pixel values
    return frame

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Make predictions
    predictions = model.predict(np.expand_dims(processed_frame, axis=0))
    predicted_class = np.argmax(predictions)

    # Display the predicted class on the frame
    cv2.putText(frame, f'Predicted: {class_labels[predicted_class]}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Sign Language Translator', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
        break

cap.release()
cv2.destroyAllWindows()
