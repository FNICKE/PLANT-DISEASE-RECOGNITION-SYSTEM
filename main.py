import streamlit as st
import tensorflow as tf
import numpy as np
import base64
import streamlit as st

import os

# ✅ Function to Set Background Image with Enhanced Styling
def set_background(image_path):
    if not os.path.exists(image_path):
        st.error(f"Error: Background image not found at {image_path}")
        return

    with open(image_path, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()

    page_bg_img = f"""
    <style>
    body {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #fff;
        font-family: 'Arial', sans-serif;
    }}

    /* Sidebar */
    .css-1d391kg {{
        background: rgba(0, 0, 0, 0.85) !important;
        color: #fff;
    }}

    /* Headers & Text */
    h1, h2, h3 {{
        text-align: center;
        color: #ffcc00;
        text-shadow: 3px 3px 5px rgba(0,0,0,0.5);
        font-size: 40px;
    }}
    p, li {{
        font-size: 20px;
    }}

    /* Buttons */
    .stButton > button {{
        background-color: #ffcc00;
        color: black;
        border-radius: 10px;
        font-size: 22px;
        font-weight: bold;
        width: 100%;
        padding: 12px;
    }}

    .stButton > button:hover {{
        background-color: #ffaa00;
        transform: scale(1.08);
        transition: 0.3s ease-in-out;
    }}

    /* Image Display */
    .stImage > img {{
        display: block;
        margin: auto;
        border-radius: 10px;
        border: 4px solid #ffcc00;
    }}

    /* Prediction Box */
    .prediction-box {{
        text-align: center;
        background-color: #ffcc00;
        padding: 18px;
        border-radius: 12px;
        font-size: 28px;
        font-weight: bold;
        margin-top: 20px;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ✅ Updated Background Image Path
bg_image_path = r"C:\Users\SACHIN RATHOD\Downloads\Jupyter Projects\machineLearning-main\machineLearning-main\Plant_Disease_Prediction\image copy.png"
set_background(bg_image_path)

# ✅ TensorFlow Model Prediction Function
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# ✅ Sidebar Navigation
st.sidebar.title("🌿 Dashboard")
app_mode = st.sidebar.radio("📌 Select Page", ["🏠 Home", "ℹ️ About", "🔍 Disease Recognition"])

# ✅ Home Page (NO CHANGE)
if app_mode == "🏠 Home":
    st.title("🌱 PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    ## 🌿 Welcome!
    **Identify plant diseases instantly!**  
    Upload an image and let our AI-powered system detect potential diseases with **high accuracy**. 🌱  

    ### ✨ Why Use This?
    - 🚀 **Fast & Efficient**
    - 🎯 **Accurate Predictions**
    - 🌎 **User-Friendly Interface**  

    **🔍 Get Started:** Click "Disease Recognition" in the sidebar!  
    """)

# ✅ About Page
elif app_mode == "ℹ️ About":
    st.title("📖 About")
    st.markdown("""
    ## 🌾 Dataset Information
    This dataset contains **87,000+ images** of healthy and diseased crop leaves.  
    Categorized into **38 different classes**, it helps detect diseases with high accuracy.

    **📂 Content:**
    - 📌 Train: **70,295 images**
    - 📌 Test: **33 images**
    - 📌 Validation: **17,572 images**  
    """)

# ✅ Disease Recognition Page
elif app_mode == "🔍 Disease Recognition":
    st.title("🔍 Disease Recognition")
    test_image = st.file_uploader("📸 Upload a Plant Image:", type=["jpg", "png", "jpeg"])

    if test_image is not None:
        st.image(test_image, use_column_width=True)

        if st.button("⚡ Predict Now"):
            st.snow()
            st.success("🔍 Processing Image...")

            # ✅ Convert Uploaded Image to Path
            with open("temp_image.jpg", "wb") as f:
                f.write(test_image.getbuffer())
            result_index = model_prediction("temp_image.jpg")

            # ✅ Class Labels
            class_name = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]

            # ✅ Stylish Prediction Output
            st.markdown(f"""
            <div class="prediction-box">
            ✅ Model Prediction: <b>{class_name[result_index]}</b>
            </div>
            """, unsafe_allow_html=True)
