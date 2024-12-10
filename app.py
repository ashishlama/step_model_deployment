from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import pickle
import json
import logging
# from sklearn.preprocessing import OneHotEncoder
import pandas as pd
# import tensorflow as tf
import os
# from werkzeug.utils import secure_filename
# from tensorflow.keras.models import load_model
# import h5py

app = Flask(__name__)

# Function to load configuration file
def load_config(config_file='config.json'):
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


# def fix_model(model_path):
#     with h5py.File(model_path, 'r+') as f:
#         model_config = f.attrs['model_config'].decode('utf-8')
#         model_config = model_config.replace('"batch_shape": [', '"batch_input_shape": [')
#         f.attrs['model_config'] = model_config.encode('utf-8')

def get_model_paths(config):
    keras_model_paths = config.get('keras_model_paths', [])
    keras_model_weights_path = config.get('keras_model_weights', '')
    pickle_model_path = config.get('pickle_model', '')
    log_path = config.get('log_path', '')
    health_risk_model_encoder_path = config.get('health_risk_model_encoder_path', '')
    return keras_model_paths, keras_model_weights_path, pickle_model_path, log_path, health_risk_model_encoder_path

# def convert_h5_to_saved_model(h5_model_path, saved_model_dir):
#     """
#     Converts a Keras .h5 model to TensorFlow SavedModel format.
#     """
#     try:
#         logger.debug(f"Loading .h5 model from {h5_model_path}")
#         model = load_model(h5_model_path)
#         logger.debug("Model loaded successfully.")
        
#         # Save as SavedModel format
#         logger.debug(f"Saving model to {saved_model_dir}")
#         model.save(saved_model_dir)
#         logger.info(f"Model converted and saved at {saved_model_dir}")
#         return saved_model_dir
#     except Exception as e:
#         logger.error(f"Failed to convert model {h5_model_path}: {e}")
#         return None
    

# def load_saved_model(model_path):
#     """
#     Loads a TensorFlow SavedModel.
#     """
#     try:
#         logger.debug(f"Loading SavedModel from {model_path}")
#         model = tf.keras.models.load_model(model_path)
#         logger.info("Model loaded successfully.")
#         return model
#     except Exception as e:
#         logger.error(f"Failed to load model from {model_path}: {e}")
#         return None
    
config = load_config()

keras_model_paths, keras_model_weights_path, pickle_model_path, log_path, health_risk_model_encoder_path = get_model_paths(config)

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

# Define allowed extensions for image upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# keras_models = []
# for path in keras_model_paths:
#     try:
#         # fix_model(path)  # Adjust model config if needed
#         keras_models.append(load_model(path))
#     except Exception as e:
#         logger.error(f"Failed to load model {path}: {e}")
# keras_models = [load_model(path) for path in keras_model_paths]
# print(f"Loaded {len(keras_models)} Keras models successfully.")

splits = [
    {'split_1': ['apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare', 
     'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito', 
     'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake', 
     'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla', 
     'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder']},
     
    {'split_2': ['club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes', 
     'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots', 
     'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries', 
     'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice', 'frozen_yogurt', 
     'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich']},
     
    {'split_3': ['grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup', 
     'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque', 
     'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup', 'mussels', 
     'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella', 'pancakes', 
     'panna_cotta', 'peking_duck']},
     
    {'split_4': ['pho', 'pizza', 'pork_chop', 'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 
     'ravioli', 'red_velvet_cake', 'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 
     'shrimp_and_grits', 'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak', 
     'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu', 'tuna_tartare']},
     
    {'split_5': ['grapes', 'peas', 'pineapple', 'turnip', 'lettuce', 'soy beans', 'spinach', 'cucumber', 'onion', 
     'cabbage', 'garlic', 'tomato', 'bell pepper', 'sweetcorn', 'capsicum', 'pear', 'beetroot', 'jalepeno', 
     'kiwi', 'chilli pepper', 'corn', 'mango', 'eggplant', 'watermelon', 'paprika', 'carrot', 'lemon', 'raddish', 
     'cauliflower', 'pomegranate', 'potato', 'banana']}
]

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Image preprocessing function (adjust if necessary)
# def preprocess_image(img_path):
#     logger.info("Preprocessing image")
#     img = tf.keras.preprocessing.image.load_img(img_path, target_size=(299, 299))  # Adjust for your model's input size
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0  # Normalize
#     logger.info("Completed preprocessing the image")
#     return img_array

# Combined model prediction function (same as before)
# def combined_model_prediction(image_path, keras_models, splits, output_path=None):
#     img_array = preprocess_image(image_path)
#     best_prediction = None
#     best_confidence = -1
#     best_class = None
#     best_split = None
#     index = 0

#     for model, split in zip(keras_models, splits):
#         try:
#             logger.info(f"Predicting using model {index + 1}")
#             logger.info(f"Prediction in progress for model {index + 1}")
#             # Get model predictions for the image
#             predictions = model.predict(img_array)
#             logger.info(f"Prediction complete for model {index + 1}")
#             # Get the highest confidence from the model's prediction
#             confidence = np.max(predictions)
#             index+=1

#             split_key = list(split.keys())[0]  # Get the key (e.g., 'split_1')
#             split_classes = split[split_key] 

#             # Update the best prediction if the confidence is higher than the previous best
#             if confidence > best_confidence:
#                 logger.info(confidence)
#                 best_confidence = confidence
#                 prediction_index = np.argmax(predictions)
#                 logger.info(f"Prediction index: {prediction_index}")
#                 best_class = split_classes[prediction_index]  # Choose the class with the highest prediction
#                 best_split = split_key
#         except Exception as e:
#             print(f"Error processing model {index + 1}: {e}")
#             continue

#     if best_class is None:
#         raise ValueError("No valid prediction was found. Check your models and data.")
    
#     print(f"Best prediction: {best_class} (confidence: {best_confidence:.2f}) from {best_split}")
#     return best_class

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

# def convertImage(file):
#     try:

#         img = Image.open(file).convert('L')  # Convert image to grayscale
        
#         # Resize the image to 28x28 pixels (you can change the size as needed)
#         img_resized = img.resize((28, 28))
        
#         # Convert image to a numpy array
#         x = np.array(img_resized)
        
#         # Invert the image (if required)
#         x = 255 - x  # Invert image to match the model's training (if needed)
        
#         # Normalize the image and reshape it to match the model's input
#         x = x.reshape(1, 28, 28, 1) / 255.0  # Normalize to [0, 1] and reshape to (1, 28, 28, 1)
        
#         # Flatten the image to (1, 784) before passing to the model
#         x = x.reshape(1, 784)  # Flatten to 784

#         return x
#     except Exception as e:
#         logger.error(f"Error processing the image: {e}")
#         raise

@app.route('/')
def index_view():
    return render_template('index.html')

# Endpoint to classify an image
# @app.route('/classify_image', methods=['POST'])
# def classify_image():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file provided'}), 400

#     file = request.files['file']

#     if file and allowed_file(file.filename):
#         filename = secure_filename(file.filename)
#         file_path = os.path.join('uploads', filename)
#         file.save(file_path)

#         try:
#             # Get predictions using the combined model prediction function
#             best_class = combined_model_prediction(
#                 file_path,keras_models, splits
#             )

#             os.remove(file_path)  # Clean up uploaded file

#             return jsonify({
#                 'class_label': best_class 
#                 # ,
#                 #     'confidence': float(confidence),
#                 #     'health_risk_prediction': float(health_risk),
#                 #     'best_model_index': int(best_model_index)
#             })
#         except Exception as e:
#             return jsonify({'error': str(e)}), 500
#     else:
#         return jsonify({'error': 'Invalid file type'}), 400
    

@app.route('/predict_health_risk', methods=['POST'])
def predict_health_risk():
    try:
        input_data = request.json.get("data")
        logger.info("Predicting health risk")
        input_df = pd.DataFrame([input_data])

        numeric_columns = ['Calories', 'Weight', 'Height', 'BMI', 'Waist circumference', 'hip circumference', 'Age', 'Monthly_eatingout_spending', 'Minutes_walk/bicycle', 'Minutes_doing_recreationalactivities', 'Sleep hours_weekdays', 'Sleep_hours_weekend', 'No_of_cigarettes_perday']
        categorical_columns = ['MaritalStatus', 'GenderStatus', 'Family_SizeStatus', 'Family_IncomeStatus', 'Alcohol_consumptionStatus']
        encoded_categorical_data = health_risk_model_encoder.transform(input_df[categorical_columns])
        encoded_feature_names = health_risk_model_encoder.get_feature_names_out()
        # Create DataFrame for encoded categorical data
        encoded_categorical_df = pd.DataFrame(encoded_categorical_data, columns=encoded_feature_names, index=input_df.index)

        # Ensure all expected columns are present in the encoded data
        missing_columns = set(encoded_feature_names) - set(encoded_categorical_df.columns)
        for col in missing_columns:
            encoded_categorical_df[col] = 0  # Add missing columns with default values

        # Combine encoded categorical data with numeric data
        merged_df = pd.concat([encoded_categorical_df, input_df[numeric_columns]], axis=1)

        input_data_reshaped = merged_df.values.reshape(1, -1)
        logger.info(input_data_reshaped)

        prediction = pickle_model.predict(input_data_reshaped)
        logger.info("Prediction: %s", prediction.tolist() )

        predicted_probabilities = pickle_model.predict_proba(input_data_reshaped)[:, 1]

        health_risk_score = predicted_probabilities * 100  # Scaled to 0-100 for interpretation
        # Round the health risk scores to 2 decimal places
        health_risk_score_rounded = [float(round(score, 0)) for score in health_risk_score]
        logger.info("Health Risk Scores: %s", health_risk_score_rounded)

        logger.info("Prediction successful for Health risk model.")
        return jsonify({"Health_risk_score": health_risk_score_rounded[0]})

    except Exception as e:
        logger.error("Error during Health risk model prediction: %s", str(e))
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    logger.info("Starting Flask app...")
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True, port=8000)