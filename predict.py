from pathlib import Path
import sys

import numpy as np
import tensorflow as tf


# This is the image size used by MobileNetV2.
IMAGE_SIZE = 224

# Resolve paths relative to this script so prediction works from any folder.
BASE_DIR = Path(__file__).resolve().parent

# These class names must be in the same order as train.py.
CLASS_NAMES = ["Normal Road", "Pothole"]

# This is the saved model file name.
MODEL_PATH = BASE_DIR / "pothole_mobilenetv2_model.keras"

# Load the trained model.
if not MODEL_PATH.exists():
    print(f"Model not found: {MODEL_PATH}")
    print("Run train.py first to create pothole_mobilenetv2_model.keras.")
    sys.exit(1)

model = tf.keras.models.load_model(MODEL_PATH)

# Ask the user to enter the image path.
image_path = input("Enter image path: ")

# Remove extra spaces and quotes from the image path.
image_path = image_path.strip().strip('"')

# Check if the image path exists and is a file.
image_file = Path(image_path)

if not image_file.exists():
    print("Image not found")
    sys.exit(1)

if not image_file.is_file():
    print("Please enter a path to an image file, not a folder.")
    exit()

# Read the image from the path.
image = tf.io.read_file(str(image_file))

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

