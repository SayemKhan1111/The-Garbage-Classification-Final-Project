import gradio as gr
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

# 🔄 Load the trained model
model = tf.keras.models.load_model("models/efficientnetv2b2_model.keras")

# 🏷️ Class names (same order as your dataset folders)
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 🔍 Prediction function
def classify_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0]
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    return f"Predicted: {pred_class} (Confidence: {confidence:.2f})"

# 🌐 Gradio Interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),  # Removed 'live=True'
    outputs="text",
    title="Garbage Classifier",
    description="Upload or use webcam to classify garbage images."
)

iface.launch(share=True)
