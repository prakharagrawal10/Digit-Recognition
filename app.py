from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import io
import base64

# Load your pre-trained model
from keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('my_model_digitswith_augmentation.h5')

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image data from the request
    data = request.get_json()
    image_data = data['image'][22:]  # Remove data URL prefix
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Preprocess the image
    img = image.resize((28, 28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255.0
    
    # Make prediction
    prediction = model.predict(img)
    digit = np.argmax(prediction)
    
    return jsonify({'digit': str(digit)})

if __name__ == '__main__':
    app.run(debug=True)
