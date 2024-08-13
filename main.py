from tensorflow.keras.models import load_model
from helper import classify_emotion
from flask import Flask, request, jsonify, render_template, url_for
import os


model_path = './best_model.h5'
model = load_model(model_path)

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        file_path = os.path.join('static', 'uploaded_image.jpg')
        file.save(file_path)
        emotion = classify_emotion(file_path)

        return render_template('index.html', image_path=url_for('static', filename='uploaded_image.jpg'), emotion=emotion)

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
