from os import write
import streamlit as st
import numpy as np
import torch
from torch import nn
from torch._higher_order_ops.while_loop import while_loop_fake_tensor_mode
import torchvision.transforms as transforms
import cv2
from PIL import Image
import time
import zipfile
import random
import io

# Streamlit Page Configuration
st.set_page_config(page_title="Alzheimer's Disease Detection", layout="wide")

# Set manual seed for reproducibility
torch.manual_seed(42)


# Image Preprocessor
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.Resize((128, 128)),  # Change to match your model input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.27777689695358276], std=[0.31906041502952576 ])  # Adjust based on dataset
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Load Model
@st.cache_resource
def load_model():
    class CNNModel(nn.Module):
        def __init__(self, num_classes=4):
            super().__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.SiLU(), nn.BatchNorm2d(32),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.SiLU(), nn.BatchNorm2d(64), nn.Dropout(0.3),
                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.SiLU(), nn.BatchNorm2d(128), nn.Dropout(0.4),
                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.SiLU(), nn.BatchNorm2d(256), nn.Dropout(0.5),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1), nn.SiLU(), nn.BatchNorm2d(512), nn.Dropout(0.5),
                nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), nn.SiLU(), nn.BatchNorm2d(512), nn.Dropout(0.5),
                nn.AdaptiveAvgPool2d((1, 1))
            )
            self.fc = nn.Linear(512, num_classes)
        def forward(self, x):
            x = self.conv_layers(x)
            x = torch.flatten(x, 1)
            return self.fc(x)

    model = CNNModel(num_classes=4)
    model.load_state_dict(torch.load("CNN_Final.pth", weights_only=True))
    model.eval()
    return model

model = load_model()

# GRAD CAM
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Hook to get gradients
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        # Hook to get feature maps
        def forward_hook(module, input, output):
            self.activations = output

        target_layer.register_forward_hook(forward_hook)
        #target_layer.register_backward_hook(backward_hook)            don't use this 
        target_layer.register_full_backward_hook(backward_hook)


    def generate_cam(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backpropagate for the specific class
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute Grad-CAM
        gradients = self.gradients.cpu().data.numpy()[0]  # Gradients from the target class
        
        #gradients2 = np.mean(gradients, axis=(1, 2))  # Example: Averaging across spatial dimensions
        #print("Gradient mean:", gradients)
        
        activations = self.activations.cpu().data.numpy()[0]  # Feature maps
        weights = np.mean(gradients, axis=(1, 2))  # Global Average Pooling

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = np.maximum(cam, 0)  # ReLU
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))  # Resize to input size
        eps = 1e-8  # Small constant to prevent division by zero
        denom = np.max(cam) - np.min(cam)
        if denom == 0:
            #print("Warning: Activation map has zero variance. Assigning a neutral heatmap.")
            cam = np.zeros_like(cam)  # Neutral output
        else:
            cam = (cam - np.min(cam)) / (denom + eps)  # Normalize safely
        return cam

# Overlay Heatmap over the input image
def overlay_heatmap(image, cam, alpha=0.5):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlayed


# Load model and define Grad-CAM
target_layer = model.conv_layers[15]  # Change this to the last convolutional layer of your model
grad_cam = GradCAM(model, target_layer)


# generating the grad cam of the input image
def grad_cam_of_input(image):
    
    input_tensor = preprocess_image(image)
    
    # Generate Grad-CAM
    cam = grad_cam.generate_cam(input_tensor)
        
    # Overlay the heatmap on the original image
    overlayed_image = overlay_heatmap(np.array(image), cam)
    
    return overlayed_image


# Random image selector
#@st.cache_resource            don't use this as it will return the same image every time
def random_image_selection():
    folders = ["Non_Demented", "Very_Mild_Demented", "Mild_Demented", "Moderate_Demented"]
    folder_image_names = ["non", "verymild", "mild", "moderate"]
    i = random.randint(0, 3)
    folder, folder_image_name = folders[i], folder_image_names[i]
    folder_sizes = [500, 2280, 916, 64]
    image_no = random.randint(2, folder_sizes[i] - 1)
    zip_path = rf"../../Dataset/{folder}.zip"
    target_image_name = f"{folder_image_name}_{image_no}.jpg"
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        with zip_file.open(f"{folder}/{target_image_name}") as image_file:
            return Image.open(io.BytesIO(image_file.read()))


prediction_classes = ["Non-Demented", "Very Mildly Demented", "Mildly Demented", "Moderately Demented"]

# Prediction and confidence score calculator
def predict_and_calculate_confidence(image):
    image = preprocess_image(image)
    output = model(image)
    output = torch.softmax(output, dim=1)
    confidence_score = torch.max(output).item()
    predicted_class_num = torch.argmax(output).item()
    predicted_class = prediction_classes[predicted_class_num]
    if confidence_score < 0.5:
        confidence_score = 1 - confidence_score
    
    return predicted_class, confidence_score, predicted_class_num


# Grad cam image
grad_cam_image = Image.open(r'grad_cam_plot_1000_dpi.png')

explanations = [
    """
    - No significant brain atrophy (shrinkage).  
    - Hippocampus and cortex maintain normal volume.  
    - No visible enlargement of ventricles (fluid-filled spaces in the brain).  
    - Clear, well-defined gray and white matter boundaries.  
    - Normal thickness of the cerebral cortex.
    """,
    
    """
    - Subtle atrophy in the hippocampus and entorhinal cortex.  
    - Slight widening of the sulci (grooves on the brain surface).  
    - Minimal increase in ventricle size.  
    - White matter may show tiny hyperintensities (early signs of vascular damage).
    """,
    
    """
    - Noticeable shrinkage of the hippocampus, particularly in the medial temporal lobe.  
    - More prominent widening of the sulci, especially in the parietal and temporal lobes.  
    - Ventricles appear larger due to surrounding brain atrophy.  
    - Reduced gray matter volume, particularly in areas involved in memory and cognition.  
    - White matter lesions (small bright spots in T2-weighted MRI) may start appearing. 
    """,
    
    """
    - Significant atrophy in the **hippocampus, temporal lobes, and parietal lobes**.  
    - Large **ventricular enlargement** (particularly in the lateral ventricles).  
    - Deepening of sulci, giving the brain a more "shrunken" appearance. 
    - Considerable gray matter loss, especially in the prefrontal cortex. 
    """
    
]


# -------- HEADER --------
st.markdown(
    "<h1 style='text-align: center;'>🧠 Alzheimer's Disease Detection</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h4 style='text-align: center;'> by Geetha Shashank Pericherla | "
    "<a href='https://github.com/Shashank-Pericherla/Alzheimers-Detection-ML/' target='_blank'><b>Repository Link</b></a> </h4>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: #FFA500; font-size: 14px;'>"
    "⚠ This tool is for educational purposes only and should not be used as a substitute for a professional medical diagnosis of Alzheimer's disease."
    "</p>",
    unsafe_allow_html=True
)





st.write("")

# -------- LAYOUT --------
col1, col2 = st.columns([1, 1])  

# -------- LEFT COLUMN (Alzheimer's Disease Info) --------
with col1:
    st.subheader("📌 About Alzheimer's Disease")
    st.markdown("""
    - Alzheimer's is a type of **dementia** that leads to progressive neurodegeneration.
    - It causes **brain atrophy, vascular shrinkage, and structural damage** over time.
    - Symptoms include **memory loss, cognitive decline, and behavioral changes** due to nerve cell degeneration.
    - Learn more about Alzheimer's disease:
    - 📖 **[Alzheimer's Association](https://www.alz.org/)**
    - 🔗 **[NIH Resource](https://www.nia.nih.gov/health/alzheimers)**
    """)

# -------- RIGHT COLUMN (Model Overview) --------
with col2:
    st.subheader("🖥️ Model Overview")
    st.markdown("""
    - **Custom CNN Model** with **6 convolutional layers** for MRI classification.
    - **Feature Extraction:** Detects key **spatial patterns** like tissue density changes.
    - **Classification:** Extracted features pass through **fully connected layers** to predict **Alzheimer’s stage**.
    - **Trained on Axial MRI Dataset** to ensure **accurate predictions**.
    - **Achieves 97.2% accuracy with impressive precision, recall scores along with near zero calibration error and 99% AUC score** on test set.
    - 📌 **For More Details:** Visit GitHub Repository🔗
    """)

st.markdown("---")

# -------- INPUT SECTION --------
col1, col2, col3, col4 = st.columns([1, 0.5, 1,1])  # Adjust width proportions

with col1:
    st.subheader("📥 Upload or Select Image")
    uploaded_image = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])
    st.markdown("<h5 style='text-align: center;'> OR </h5>", unsafe_allow_html=True)
    st.write("If you don't have an image, you can select a random image from the test set.")
    
    if st.button("Random Image Generator"):
        if "selected_image" in st.session_state:
            del st.session_state.selected_image  # Ensure old image is cleared
        if "prediction" in st.session_state:
            del st.session_state.prediction  # Ensure old prediction is cleared
        st.session_state.selected_image = random_image_selection()
    
    # Store uploaded image in session state
    if uploaded_image is not None:
        if "selected_image" in st.session_state:
            del st.session_state.selected_image  # Ensure old image is cleared
        if "prediction" in st.session_state:
            del st.session_state.prediction  # Ensure old prediction is cleared
        
        st.session_state.selected_image = Image.open(uploaded_image)
    
    if "selected_image" in st.session_state:
        st.image(st.session_state.selected_image, caption="Selected Image", use_container_width=False, width= 128)

# -------- PREDICTION Button --------
predicted_class = None

# CSS for centering and styling the button
st.markdown(
    """
    <style>
    .centered-button {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;  /* Ensures vertical centering */
    }
    
    .stButton>button {
        width: 200px !important;  /* Adjust width */
        height: 50px !important;  /* Adjust height */
        font-size: 18px !important;  /* Increase text size */
        background-color: #003366 !important;  /* Dark Blue */
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)


with col2:
    st.subheader("")  # Keeps space for alignment
    st.subheader("")
    st.write("")  # Extra spacing if needed
    st.write("")
    

    # Add a div for centering
    st.markdown('<div class="centered-button">', unsafe_allow_html=True)
    
    if st.button("Predict"):
        if "selected_image" in st.session_state:
            prediction_class, confidence_score, predicted_class_num = predict_and_calculate_confidence(st.session_state.selected_image)
            st.session_state.prediction = (prediction_class, confidence_score, predicted_class_num)
            
    st.markdown('</div>', unsafe_allow_html=True)  # Close the div
    





# -------- OUTPUT SECTION --------
with col3:
    st.subheader("📊 Prediction Output")
    st.write("")
    st.write("")
    if "prediction" in st.session_state:
        prediction_class, confidence_score, predicted_class_num = st.session_state.prediction
        
        uploaded_image = st.session_state.selected_image
        output_grad_image = grad_cam_of_input(uploaded_image)
        
        time.sleep(2)
        st.write(f"### Prediction: {prediction_class}")
        st.write(f"### Confidence: {confidence_score:.2f}%")
        st.write("")
        st.write("")
        
        st.image(output_grad_image, caption="Gradient Focus", use_container_width=False, width= 128)
        
        

# -------- NEXT ROW: TEXT CONTENT --------

with col4:
    st.subheader("Explanation for the Model's Prediction:")

    if "prediction" in st.session_state:
        prediction_class, confidence_score, predicted_class_num = st.session_state.prediction
        st.write(explanations[predicted_class_num])


st.markdown("---")

# -------- NEXT ROW: GRAD-CAM IMAGE --------

col5 , col6 = st.columns([0.6, 1])

with col6:
    st.subheader("📸 Grad-CAM Heatmap:")
    st.write("This heatmap shows where the model focuses its attention at each convolution layer.")
    st.image(grad_cam_image, caption="Model's Attention Heatmap", use_container_width =True)


# -------- NEXT ROW: EXPLANATION --------
with col5:
    st.write(
        """
        ## **Key MRI Features Across Progression**  

        | Feature            | Non-Demented | Very Mild Dementia | Mild Dementia | Moderate Dementia |
        |------------------|--------------|-----------------|--------------|----------------|
        | **Hippocampus Size** | Normal | Slightly reduced | Noticeably smaller | Significantly shrunken |
        | **Ventricles**        | Normal | Slightly enlarged | Moderately enlarged | Very large |
        | **Sulci Width**       | Normal | Mild widening | More pronounced widening | Severe deepening |
        | **Gray Matter Loss**  | None | Subtle | Moderate, especially in temporal lobes | Widespread loss |
        | **White Matter Lesions** | None | Few | Visible in T2 MRI | Extensive |
        | **Cortical Thickness** | Normal | Slight thinning | Reduced | Markedly reduced |
        """
    )