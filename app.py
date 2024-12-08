from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import base64
import re
import pickle
from tensorflow.keras.models import model_from_json
import json
import logging
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

app = Flask(__name__)

# Function to load configuration file
def load_config(config_file='config.json'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def get_model_paths(config):
    keras_model_architecture_path = config.get('keras_model_architecture', '')
    keras_model_weights_path = config.get('keras_model_weights', '')
    pickle_model_path = config.get('pickle_model', '')
    log_path = config.get('log_path', '')
    health_risk_model_encoder_path = config.get('health_risk_model_encoder_path', '')
    return keras_model_architecture_path, keras_model_weights_path, pickle_model_path, log_path, health_risk_model_encoder_path

config = load_config()

keras_model_architecture_path, keras_model_weights_path, pickle_model_path, log_path, health_risk_model_encoder_path = get_model_paths(config)

log_filename = 'app.log'
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  
    ]
)

logger = logging.getLogger(__name__)

logger.info("Loading TensorFlow model architecture from JSON...")
with open(keras_model_architecture_path, 'r') as json_file:
    model_json = json_file.read()

keras_model = model_from_json(model_json)

logger.info("Loading TensorFlow model weights from .h5 file...")
keras_model.load_weights(keras_model_weights_path)

logger.info("Compiling the TensorFlow model...")
keras_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def load_pickle_model():
    logger.info("Loading Pickle model from .pkl file...")
    with open(pickle_model_path, 'rb') as pickle_file:
        pickle_model = pickle.load(pickle_file)
    logger.info("Pickle model loaded successfully.")
    return pickle_model

def load_model_encoder():
    logger.info("Loading Encoder of model from .pkl file...")
    with open(health_risk_model_encoder_path, 'rb') as encoder_pickle_file:
        encoder_pickle_model = pickle.load(encoder_pickle_file)
    logger.info("Encoder of model loaded successfully.")
    return encoder_pickle_model

# Preprocessing function for the input data for Keras model
def preprocess_input(data):
    logger.debug(f"Preprocessing input data: {data}")
    # Example: Convert image to grayscale and resize to the model's input size
    data = np.array(data, dtype=np.float32)
    data = np.expand_dims(data, axis=0)  # Add batch dimension
    data = data / 255.0  # Normalize to [0, 1]
    logger.debug(f"Processed input data: {data}")
    return data

pickle_model = load_pickle_model()
health_risk_model_encoder = load_model_encoder()

# Function to convert base64 image to an actual image file
def convertImage(imgData1):
    imgstr = re.search(b'base64, (.*)', imgData1).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))

@app.route('/')
def index_view():
    return render_template('index.html')

@app.route('/classify_image', methods=['POST'])
def classify_image():
    try:
        # Parse input data (assuming base64-encoded image in the request body)
        imgData = request.get_data()
        convertImage(imgData)
        logger.info("Classifying images")
        # Open the image, convert it to grayscale and resize it
        x = Image.open('output.png').convert('L')  # Convert image to grayscale
        x = np.array(x.resize((28, 28)))  # Resize to 28x28 (change based on your model)
        x = 255 - x  # Invert the image
        x = x.reshape(1, 28, 28, 1) / 255.0  # Normalize to [0, 1] and reshape

        # Preprocess the image
        processed_data = preprocess_input(x)

        # Make the prediction using the Keras model
        predictions = keras_model.predict(processed_data)

        # Return the prediction result as JSON
        response = np.argmax(predictions, axis=1)
        logger.info("Classification successful for model.")
        return jsonify({"predictions": response.tolist()})

    except Exception as e:
        logger.error("Error during classification for model: %s", str(e))
        return jsonify({"error": str(e)}), 400

@app.route('/predict_health_risk', methods=['POST'])
def predict_health_risk():
    try:
        input_data = request.json.get("data")
        logger.info("Predicting health risk")
        input_df = pd.DataFrame([input_data])

        encoded_input_data = health_risk_model_encoder.transform(input_df)
        feature_names = health_risk_model_encoder.get_feature_names_out()

        encoded_input_df = pd.DataFrame(encoded_input_data, columns=feature_names, index=input_df.index)

        # Make sure the input data has the same columns as the model expects
        # (adding missing columns if necessary, as the encoder might ignore some unknown values)
        missing_columns = set(health_risk_model_encoder.get_feature_names_out()) - set(encoded_input_df.columns)
        for col in missing_columns:
            encoded_input_df[col] = 0 

        encoded_input_df = encoded_input_df[health_risk_model_encoder.get_feature_names_out()]

        input_data_reshaped = encoded_input_df.values.reshape(1, -1)

        prediction = pickle_model.predict(input_data_reshaped)

        logger.info("Prediction successful for Health risk model.")
        return jsonify({"predictions": prediction.tolist()})

    except Exception as e:
        logger.error("Error during Health risk model prediction: %s", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    app.run(debug=True, port=8000)