from pathlib import Path

import numpy as np
import tensorflow as tf


# This is the image size used by MobileNetV2.
IMAGE_SIZE = 224

# These class names must be in the same order as train.py.
CLASS_NAMES = ["Normal Road", "Pothole"]

# This is the saved model file name.
MODEL_PATH = "pothole_mobilenetv2_model.keras"

# Load the trained model.
model = tf.keras.models.load_model(MODEL_PATH)

# Ask the user to enter the image path.
image_path = input("Enter image path: ")

# Remove extra spaces and quotes from the image path.
image_path = image_path.strip().strip('"')

# Check if the image path exists.
if not Path(image_path).exists():
    print("Image not found")
    exit()

# Read the image from the path.
image = tf.io.read_file(image_path)

# Decode jpg, jpeg, or png image into 3 color channels.
image = tf.io.decode_image(image, channels=3, expand_animations=False)

# Resize the image to 224 x 224.
image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

# Convert the image into a numpy array.
image = image.numpy()

# Add one extra batch dimension because the model expects a batch of images.
image = np.expand_dims(image, axis=0)


# Predict class probabilities.
prediction = model.predict(image, verbose=0)

# Get the class number with highest probability.
class_number = np.argmax(prediction[0])

# Get the confidence value.
confidence = prediction[0][class_number] * 100

# Print final prediction.
print("Prediction :", CLASS_NAMES[class_number])

# Print confidence percentage.
print("Confidence :", f"{confidence:.2f}%")

