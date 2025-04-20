import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('AI ML/Image Classification/model/mobilenet_fish_model_wo_animalfish.h5')

model = load_model()

# Assuming you have the same class names as during training
class_names = ['fish sea_food black_sea_sprat', 'fish sea_food gilt_head_bream','fish sea_food hourse_mackerel','fish sea_food red_mullet','fish sea_food red_sea_bream','fish sea_food sea_bass','fish sea_food shrimp','fish sea_food striped_red_mullet','fish sea_food trout']  # Replace with your actual class names

# Format class names for better display
formatted_classes = [cls.replace("fish sea_food ", "").replace("_", " ").title() for cls in class_names]

st.title("🐟 Sea Food Fish Classifier")
st.write("Upload a fish image to predict its species.")

st.markdown("#### Supported Fish Categories for Prediction:")
st.write(", ".join(formatted_classes))

uploaded_file = st.file_uploader("Choose a fish image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    
    # Preprocess
    img = img.resize((256, 256))  # Match input size used in training
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)

    # Example: cleaning predicted label
    predicted_class = predicted_class.replace("fish sea_food ", "").replace("_", " ").title()

    st.success(f"🎯 Predicted: {predicted_class} ({confidence*100:.2f}% confidence)")

    # st.image(img, caption='Uploaded Fish Image', use_container_width=True)

    # # Display all confidence scores
    # st.subheader("Confidence Scores:")
    # for i, score in enumerate(preds[0]):
    #     st.write(f"{formatted_classes[i]}: {score*100:.2f}%")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(img, caption='Uploaded Fish Image', use_container_width=True)

    with col2:
        st.subheader("Confidence Scores:")
        max_idx = int(np.argmax(preds[0]))
        for i, score in enumerate(preds[0]):
            label = formatted_classes[i]
            percent = score * 100
            if i == max_idx and percent > 50:
                st.markdown(f"{label}: **{percent:.2f}% 🟢**")
            else:
                st.markdown(f"{label}: **{percent:.2f}%**")
