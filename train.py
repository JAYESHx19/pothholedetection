import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input, Rescaling
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


# These are the basic settings used for training.
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
SEED = 42

# This is the folder where the dataset should be present.
DATASET_FOLDER = Path("dataset")

# Resolve output paths relative to this script so training works from any folder.
BASE_DIR = Path(__file__).resolve().parent

# These class names are printed in results.
CLASS_NAMES = ["Normal Road", "Pothole"]

# These are the actual folder names in the dataset.
CLASS_FOLDERS = ["Plain", "Pothole"]

# These image formats are supported by the project.
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def get_image_files(folder):
    # This list stores all image file paths from one folder.
    image_files = []

    # This loop checks every file inside the folder.
    for file_path in folder.rglob("*"):
        # This condition keeps only jpg, jpeg, and png files.
        if file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(str(file_path))

    # Return the final list of images.
    return image_files


def read_image(image_path, label):
    # Read the image file from the given path.
    image = tf.io.read_file(image_path)

    # Decode jpg, jpeg, or png image into 3 color channels.
    image = tf.io.decode_image(image, channels=3, expand_animations=False)

    # Resize the image to 224 x 224 for MobileNetV2.
    image = tf.image.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

    # Set the image shape so TensorFlow knows the size clearly.
    image.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])

    # Convert the label number into one-hot format for categorical crossentropy.
    label = tf.one_hot(label, len(CLASS_NAMES))

    # Return the image and its label.
    return image, label


# Set random seeds so the split remains the same every time.
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Check if the dataset folder exists.
if not DATASET_FOLDER.exists():
    raise FileNotFoundError("Please create a dataset folder with Pothole and Plain class images.")

# These lists will store image paths and labels.
train_paths = []
train_labels = []
valid_paths = []
valid_labels = []
test_paths = []
test_labels = []

# Check if the dataset already has train, valid, and test folders.
has_train_valid_test = (DATASET_FOLDER / "train").exists() and (DATASET_FOLDER / "valid").exists() and (DATASET_FOLDER / "test").exists()

# Check if the dataset has train and test folders only.
has_train_test = (DATASET_FOLDER / "train").exists() and (DATASET_FOLDER / "test").exists()

if has_train_valid_test:
    # This part is used when dataset/train, dataset/valid, and dataset/test already exist.
    print("Using existing train, valid, and test folders.")

    # This loop reads images from every class folder.
    for label_number, class_name in enumerate(CLASS_FOLDERS):
        # Read training images of one class.
        class_train_paths = get_image_files(DATASET_FOLDER / "train" / class_name)

        # Read validation images of one class.
        class_valid_paths = get_image_files(DATASET_FOLDER / "valid" / class_name)

        # Read testing images of one class.
        class_test_paths = get_image_files(DATASET_FOLDER / "test" / class_name)

        # Add image paths to the training list.
        train_paths.extend(class_train_paths)

        # Add labels for the training images.
        train_labels.extend([label_number] * len(class_train_paths))

        # Add image paths to the validation list.
        valid_paths.extend(class_valid_paths)

        # Add labels for the validation images.
        valid_labels.extend([label_number] * len(class_valid_paths))

        # Add image paths to the testing list.
        test_paths.extend(class_test_paths)

        # Add labels for the testing images.
        test_labels.extend([label_number] * len(class_test_paths))
elif has_train_test:
    # This part is used when dataset/train and dataset/test exist, but valid folder is missing.
    print("Using train and test folders. Creating validation data from training images.")

    # This loop reads images from every class folder.
    for label_number, class_name in enumerate(CLASS_FOLDERS):
        # Read training images of one class.
        class_train_paths = get_image_files(DATASET_FOLDER / "train" / class_name)

        # Read testing images of one class.
        class_test_paths = get_image_files(DATASET_FOLDER / "test" / class_name)

        # Shuffle training images using the fixed seed.
        random.shuffle(class_train_paths)

        # Use 90% of train folder for training.
        train_end = int(len(class_train_paths) * 0.9)

        # Add first 90% images to training data.
        train_paths.extend(class_train_paths[:train_end])

        # Add last 10% images to validation data.
        valid_paths.extend(class_train_paths[train_end:])

        # Add test folder images to testing data.
        test_paths.extend(class_test_paths)

        # Add labels for training images.
        train_labels.extend([label_number] * len(class_train_paths[:train_end]))

        # Add labels for validation images.
        valid_labels.extend([label_number] * len(class_train_paths[train_end:]))

        # Add labels for testing images.
        test_labels.extend([label_number] * len(class_test_paths))
else:
    # This part is used when dataset has only Pothole and Plain class folders.
    print("Splitting dataset into 80% train, 10% validation, and 10% test.")

    # This loop reads both class folders and splits them separately.
    for label_number, class_name in enumerate(CLASS_FOLDERS):
        # Read all images of one class.
        class_paths = get_image_files(DATASET_FOLDER / class_name)

        # Shuffle images using the fixed seed.
        random.shuffle(class_paths)

        # Count total images in this class.
        total_images = len(class_paths)

        # Calculate 80% position for training data.
        train_end = int(total_images * 0.8)

        # Calculate 90% position, so 80-90 is validation and 90-100 is test.
        valid_end = int(total_images * 0.9)

        # Add first 80% images to training data.
        train_paths.extend(class_paths[:train_end])

        # Add next 10% images to validation data.
        valid_paths.extend(class_paths[train_end:valid_end])

        # Add last 10% images to testing data.
        test_paths.extend(class_paths[valid_end:])

        # Add labels for training images.
        train_labels.extend([label_number] * len(class_paths[:train_end]))

        # Add labels for validation images.
        valid_labels.extend([label_number] * len(class_paths[train_end:valid_end]))

        # Add labels for testing images.
        test_labels.extend([label_number] * len(class_paths[valid_end:]))

# Stop training if dataset is not arranged properly.
if len(train_paths) == 0 or len(valid_paths) == 0 or len(test_paths) == 0:
    raise ValueError("Dataset must contain images inside Pothole and Plain folders.")

# Print the number of images used in each split.
print("Training images:", len(train_paths))
print("Validation images:", len(valid_paths))
print("Testing images:", len(test_paths))

# Create TensorFlow training dataset from image paths and labels.
train_data = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))

# Read, resize, shuffle, and batch the training images.
train_data = train_data.map(read_image).shuffle(1000, seed=SEED).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Create TensorFlow validation dataset.
valid_data = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))

# Read, resize, and batch the validation images.
valid_data = valid_data.map(read_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Create TensorFlow test dataset.
test_data = tf.data.Dataset.from_tensor_slices((test_paths, test_labels))

# Read, resize, and batch the test images.
test_data = test_data.map(read_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Data augmentation helps the model learn from slightly changed images.
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])

# Create the input layer for 224 x 224 color images.
inputs = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# Apply data augmentation only during training.
x = data_augmentation(inputs)

# Scale image pixels from 0 to 255 into the range -1 to 1 for MobileNetV2.
x = Rescaling(1.0 / 127.5, offset=-1)(x)

# Load MobileNetV2 with ImageNet weights and remove its original top layer.
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# Freeze MobileNetV2 layers because we are using transfer learning.
base_model.trainable = False

# Pass images through the frozen MobileNetV2 model.
x = base_model(x, training=False)

# Convert feature maps into one feature vector.
x = GlobalAveragePooling2D()(x)

# Dropout helps reduce overfitting.
x = Dropout(0.3)(x)

# Final layer gives probability for Normal and Pothole.
outputs = Dense(len(CLASS_NAMES), activation="softmax")(x)

# Create the final model.
model = Model(inputs, outputs)

# Compile the model with Adam optimizer and categorical crossentropy loss.
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Show the model structure.
model.summary()

# Train the model.
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=EPOCHS,
)

# Save the trained model.
model.save(BASE_DIR / "pothole_mobilenetv2_model.keras")

# Plot training and validation accuracy.
plt.figure()
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy.png")
plt.show()

# Plot training and validation loss.
plt.figure()
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Loss vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss.png")
plt.show()

# Predict classes for test images.
predictions = model.predict(test_data)

# Convert prediction probabilities into class numbers.
predicted_labels = np.argmax(predictions, axis=1)

# Convert true test labels into a numpy array.
true_labels = np.array(test_labels)

# Calculate accuracy.
accuracy = accuracy_score(true_labels, predicted_labels)

# Calculate precision.
precision = precision_score(true_labels, predicted_labels, average="weighted", zero_division=0)

# Calculate recall.
recall = recall_score(true_labels, predicted_labels, average="weighted", zero_division=0)

# Calculate F1 score.
f1 = f1_score(true_labels, predicted_labels, average="weighted", zero_division=0)

# Print evaluation results.
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Print classification report.
print("\nClassification Report:")
print(classification_report(true_labels, predicted_labels, target_names=CLASS_NAMES, zero_division=0))

# Create confusion matrix.
cm = confusion_matrix(true_labels, predicted_labels)

# Display confusion matrix as a heatmap.
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()




