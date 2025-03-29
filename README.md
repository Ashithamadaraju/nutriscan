import streamlit as st
import requests
import tensorflow as tf
import numpy as np
from PIL import Image
import openai

# Load Pre-trained MobileNetV2 Model
model = tf.keras.applications.MobileNetV2(weights='imagenet')
labels_path = tf.keras.utils.get_file("imagenet_class_index.json","https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json")
import json
with open(labels_path) as f:
    class_labels = json.load(f)

# OpenAI API Key (Replace with your own API key)
OPENAI_API_KEY = "your-openai-api-key"
openai.api_key = OPENAI_API_KEY

# Function to Predict Food Item
def predict_food(image):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    predictions = model.predict(img_array)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=1)[0]
    return decoded_predictions[0][1]  # Return top predicted label

# Function to Fetch Nutrition Data from OpenFoodFacts API
def get_nutrition_data(food_name):
    url = f"https://world.openfoodfacts.org/api/v0/product/{food_name}.json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'product' in data and 'nutriments' in data['product']:
            return data['product']['nutriments']
    return "Nutritional data not found."

# Function to Generate Health Tips Using OpenAI
def get_health_tips(food_name):
    prompt = f"Provide a detailed nutritional analysis and health impact of {food_name}. Also, suggest healthy alternatives if necessary."
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a nutrition expert."},
                  {"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"].strip()

# Streamlit UI
st.title("NutriScan - AI Food Nutrition Scanner")
uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Analyzing...")
    
    # Predict Food Item
    predicted_food = predict_food(image)
    st.write(f"### Predicted Food: {predicted_food}")
    
    # Fetch Nutritional Data
    nutrition_info = get_nutrition_data(predicted_food)
    st.write("### Nutritional Information:")
    st.json(nutrition_info)

    # Generate Health Tips
    health_tips = get_health_tips(predicted_food)
    st.write("### AI-Generated Health Tips:")
    st.write(health_tips)

