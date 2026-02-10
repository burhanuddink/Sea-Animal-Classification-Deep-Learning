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
    
    # 1. Resize to EfficientNet standard size (224x224)
    image = image.resize((224, 224))
    
    # 2. Convert to numpy array
    img_array = np.array(image)
    
    # 3. FIX: REMOVE MANUAL RESCALING
    # The model has a Rescaling(1./255) layer built-in.
    # We pass raw pixels (0-255).
    # img_array = img_array.astype("float32") / 255.0  <-- DELETE THIS LINE
    
    # 4. Expand dimensions (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # 5. Make Prediction
    predictions = model.predict(img_array)
    
    # 6. FIX: HANDLING SOFTMAX
    # Most Keras models already have softmax in the last layer.
    # Applying it twice makes high confidence (99%) drop to low confidence (15%).
    
    # We check: If the sum is close to 1, it's already a probability.
    if np.sum(predictions[0]) > 1.1: 
        # It's logits (raw scores), so we apply softmax
        scores = tf.nn.softmax(predictions[0])
    else:
        # It's already probabilities, just use them directly
        scores = predictions[0]
    
    # Return top 3 predictions
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
