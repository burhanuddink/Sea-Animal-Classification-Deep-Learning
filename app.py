import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Loading pre-trained model
model_path = "models/COP508FinalEfficientNetBest.keras"
model = tf.keras.models.load_model(model_path)

# 2. Defining the class names
class_names = [ 
    'Corals', 'Crabs', 'Dolphin', 'Eel', 'Jelly Fish', 'Lobster', 
    'Nudibranchs', 'Octopus', 'Puffers', 'Sea Rays', 'Seahorse', 
    'Seal', 'Sharks', 'Squid', 'Starfish', 'Whale'
]

# 3. Creating a prediction function
def classify_image(image):
    if image is None:
        return None
    
    # A. Resizing image to EfficientNet standard size (224x224)
    image = image.resize((224, 224))
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    scores = tf.nn.softmax(predictions[0])
    
    # Returning top 3 predictions
    return {class_names[i]: float(scores[i]) for i in range(len(class_names))}

# 4. Building Gradio interface
interface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", label="Upload Sea Animal Image"),
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="Marine Species Classifier",
    description="Upload an image to classify it into one of 16 marine species using EfficientNetB0.",
    examples=[] 
)

# 5. Launching the App
if __name__ == "__main__":
    interface.launch()
