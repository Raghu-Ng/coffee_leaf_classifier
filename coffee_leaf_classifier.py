import os
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model('best_coffee_leaf_model_final.keras')

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
        if file:
            image = Image.open(file.stream)
            preprocessed_image = preprocess_image(image)
            prediction = model.predict(preprocessed_image)
            class_names = ['healthy', 'red_spider_mite', 'rust']
            result = class_names[np.argmax(prediction)]
            confidence = float(np.max(prediction))
            return jsonify({'prediction': result, 'confidence': confidence})
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)