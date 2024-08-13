from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
import os
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomRotation, RandomZoom, RandomFlip, GaussianNoise
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import load_model

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

#Defined functions
import numpy as np
def plot_history(history):
  plt.figure(figsize=(14, 5))
  plt.subplot(1, 2, 1)
  plt.plot(history.history['accuracy'], label='Train')
  plt.plot(history.history['val_accuracy'], label='Validation')
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(loc='upper left')

  # Plot training loss
  plt.subplot(1, 2, 2)
  plt.plot(history.history['loss'], label='Train')
  plt.plot(history.history['val_loss'], label='Validation')
  plt.title('Model loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(loc='upper left')

  plt.show()


def preprocess_image(image_path, target_size=(48, 48)):
  """Load an image and preprocess it for emotion classification."""
  img = load_img(image_path, target_size=target_size, color_mode='rgb')
  img_array = img_to_array(img)
  img_array_expanded_dims = np.expand_dims(img_array, axis=0)
  return img_array_expanded_dims


def classify_emotion(image_path, model):
  preprocessed_image = preprocess_image(image_path)
  prediction = model.predict(preprocessed_image)
  predicted_class = np.argmax(prediction)
  return emotion_labels[predicted_class]

def display_prediction(image_path):
  emotion = classify_emotion(image_path)
  img = plt.imread(image_path)
  plt.imshow(img)
  plt.title(f'Predicted Emotion: {emotion}')
  plt.axis('off')
  plt.show()


  