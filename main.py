from tensorflow.keras.models import load_model
from helper import display_prediction


model_path = 'C:\\Users\\Rober\\deploy\\best_model.h5'
model = load_model(model_path)
display_prediction('image.jpg')
