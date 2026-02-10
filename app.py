import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Load the trained model
# Ensure the model file is in the 'models' folder as discussed
model_path = "models/COP508FinalEfficientNetBest.keras"
model = tf.keras.models.load_model(model_path)

# 2. Define the 16 Class Names (Your specific list)
class_names = [ 
    'Corals', 'Crabs', 'Dolphin', 'Eel', 'Jelly Fish', 'Lobster', 
    'Nudibranchs', 'Octopus', 'Puffers', 'Sea Rays', 'Seahorse', 
    'Seal', 'Sharks', 'Squid', 'Starfish', 'Whale'
]

# 3. Define the Prediction Function
def classify_image(image):
    if image is None:
        return None
    
    # A. Resize to EfficientNet standard size (224x224)
    image = image.resize((224, 224))
    
    # B. Convert to numpy array
    img_array = np.array(image)
    
    # C. Preprocessing (Normalization)
    # Since you used Rescaling in training, we must normalize to [0, 1] range
    img_array = img_array.astype("float32") / 255.0
    
    # D. Expand dimensions to match model input shape (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # E. Make Prediction
    predictions = model.predict(img_array)
    scores = tf.nn.softmax(predictions[0])
    
    # Return top 3 predictions for the UI
    return {class_names[i]: float(scores[i]) for i in range(len(class_names))}

# 4. Build the Gradio Interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Sea Animal Image"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="🌊 Marine Species Classifier",
    description="Upload an image to classify it into one of 16 marine species using EfficientNetB0.",
    examples=[] # You can add specific image paths here later if you want
)

# 5. Launch the App
if __name__ == "__main__":
    interface.launch()
