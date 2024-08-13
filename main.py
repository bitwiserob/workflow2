from tensorflow.keras.models import load_model
from helper import classify_emotion
from flask import Flask, request, jsonify


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        file.save('image_saved.jpg')
        emotion = classify_emotion('image_saved.jpg')
        return jsonify({'emotion': emotion})


if __name__ == '__main__':
    app.run(debug=True)