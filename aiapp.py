import streamlit as st
import cv2
import numpy as np
import PIL.Image
import os
import io
import base64
import pandas as pd
import google.generativeai as genai
from time import sleep
from datetime import datetime

# ==============================================
# CONFIGURATION AND INITIALIZATION
# ==============================================

# Set page config
st.set_page_config(
    page_title="PIXTRACT - Image to Story",
    page_icon="🖼️",
    layout="wide"
)

# API key setup with error handling
try:
    os.environ["GOOGLE_API_KEY"] = "AIzaSyBfCZXuVisf-JVii6J1MBp-eDXZk03YL1o"  # Replace with your actual API key
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    model = genai.GenerativeModel("models/gemini-1.5-flash-latest")
except Exception as e:
    st.error(f"Failed to configure AI model: {str(e)}")
    st.stop()

# ==============================================
# IMAGE PROCESSING FUNCTIONS
# ==============================================

def validate_image(img):
    """Validate image integrity and format"""
    if img is None:
        return False
    try:
        if isinstance(img, PIL.Image.Image):
            img.verify()  # Verify PIL image integrity
            return True
        elif isinstance(img, (bytes, bytearray)):
            test_img = PIL.Image.open(io.BytesIO(img))
            test_img.verify()
            return True
        return False
    except Exception:
        return False

def process_image_for_gemini(img):
    """Convert image to format expected by Gemini API"""
    try:
        if isinstance(img, PIL.Image.Image):
            return img  # Already in correct format

        if isinstance(img, (bytes, bytearray)):
            return PIL.Image.open(io.BytesIO(img))

        raise ValueError("Unsupported image format")
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def get_base64_image(image_path):
    """Load local image as base64 string with fallback"""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return ""

# ==============================================
# AI PROCESSING FUNCTIONS
# ==============================================

def image_to_text(img):
    """Extract text details from image"""
    try:
        processed_img = process_image_for_gemini(img)
        if not processed_img:
            return "Failed to process image"

        with st.spinner("🔍 Extracting image details..."):
            response = model.generate_content(
                ["Extract comprehensive details from this image including objects, text, colors, and context. Be thorough.", processed_img],
                request_options={"timeout": 20}
            )
            return response.text
    except Exception as e:
        st.error(f"Extraction error: {str(e)}")
        return "Failed to extract details"

def image_and_query(query, img):
    """Generate response based on image and user query"""
    try:
        processed_img = process_image_for_gemini(img)
        if not processed_img:
            return "Failed to process image"

        if not query.strip():
            query = "Generate a detailed description, story, and analysis of this image."

        with st.spinner("🧠 Generating creative response..."):
            response = model.generate_content(
                [f"{query}\n\nUse the image as reference and provide a comprehensive response.", processed_img],
                request_options={"timeout": 30}
            )
            return response.text
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        return "Failed to generate response"

# ==============================================
# UI AND APPLICATION LOGIC
# ==============================================

# Load background image
img_base64 = get_base64_image("background5.jpg")

# Custom CSS with improved styling
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("data:image/png;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
        }}
        .main-title {{
            font-size: 3.5rem !important;
            color: #00E5FF !important;
            text-align: center;
            font-family: 'Arial Black', sans-serif;
            text-shadow: 2px 2px 15px rgba(0, 229, 255, 0.7);
            margin-bottom: 0.5rem !important;
        }}
        .subtitle {{
            font-size: 1.2rem !important;
            text-align: center;
            font-family: 'Arial', sans-serif;
            color: #B0BEC5;
            margin-bottom: 2rem !important;
        }}
        .stRadio > div {{
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 10px;
        }}
        .stButton>button {{
            background: linear-gradient(135deg, #00796B, #004D40);
            color: white;
            font-size: 1.1rem;
            border-radius: 15px;
            padding: 12px 24px;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
            width: 100%;
        }}
        .stButton>button:hover {{
            transform: scale(1.05);
            box-shadow: 0 5px 15px rgba(0, 229, 255, 0.4);
        }}
        .info-box {{
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(10px);
            padding: 25px;
            border-radius: 15px;
            margin: 2rem auto;
            text-align: center;
            max-width: 800px;
            box-shadow: 0 8px 32px rgba(0, 229, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }}
        .result-box {{
            background: rgba(0, 0, 0, 0.7);
            border-radius: 10px;
            padding: 20px;
            margin: 1rem 0;
        }}
        .image-container {{
            display: flex;
            justify-content: center;
            margin: 1.5rem 0;
        }}
        @media (max-width: 768px) {{
            .main-title {{
                font-size: 2.5rem !important;
            }}
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Main UI
st.markdown('<h1 class="main-title">🚀 PIXTRACT</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle"><b>Transform images into rich narratives with AI-powered analysis</b></p>', unsafe_allow_html=True)

# Initialize session state
if 'captured_img' not in st.session_state:
    st.session_state.captured_img = None
if 'last_processed' not in st.session_state:
    st.session_state.last_processed = None

# Image Source Selection
with st.container():
    st.markdown("### 📸 Image Source")
    image_source = st.radio(
        "Choose how to provide your image:",
        ("Upload Image", "Capture from Webcam"),
        horizontal=True,
        label_visibility="collapsed"
    )

# Image Input Section
img = None
with st.container():
    if image_source == "Upload Image":
        upload_image = st.file_uploader(
            "Upload an image (PNG, JPG, JPEG)",
            type=['png', 'jpg', 'jpeg'],
            help="Maximum file size: 5MB"
        )
        if upload_image:
            try:
                img = PIL.Image.open(upload_image)
                st.session_state.captured_img = None
                st.image(img, caption="Uploaded Image", width=400)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")

    elif image_source == "Capture from Webcam":
        img_file = st.camera_input("Take a picture")
        if img_file:
            img = PIL.Image.open(img_file)
            st.session_state.captured_img = img
            st.success("Image captured successfully!")
            st.image(img, caption="Captured Image", width=400)

# User Query Section
with st.container():
    query = st.text_area(
        "✍️ Your Prompt",
        value="",
        placeholder="Describe what you want to generate (story, analysis, etc.) or leave blank for automatic description",
        height=120,
        help="Example: 'Write a creative story based on this image' or 'Analyze the technical aspects of this photo'"
    )

# Processing Section
if st.button("✨ Generate Magic ✨", use_container_width=True, type="primary"):
    if img is not None and validate_image(img):
        with st.spinner("Working magic with your image..."):
            st.image(img, use_container_width=True)
            processed_img = process_image_for_gemini(img)

            if processed_img:
                tab1, tab2 = st.tabs(["📝 Extracted Details", "✨ Generated Content"])

                with tab1:
                    extracted_details = image_to_text(processed_img)
                    st.markdown(f'<div class="result-box">{extracted_details}</div>', unsafe_allow_html=True)

                with tab2:
                    generated_details = image_and_query(query, processed_img)
                    st.markdown(f'<div class="result-box">{generated_details}</div>', unsafe_allow_html=True)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                df = pd.DataFrame({
                    "Timestamp": [timestamp],
                    "Extracted Details": [extracted_details],
                    "Generated Content": [generated_details],
                    "User Query": [query]
                })

                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Report (CSV)",
                    data=csv,
                    file_name=f"pixstract_report_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                st.session_state.last_processed = timestamp
            else:
                st.error("Failed to process image for analysis")
    else:
        st.warning("⚠️ Please provide a valid image before generating content")

st.markdown("""
<div class="info-box">
    <h3>🎨 Key Features</h3>
    <ul style="text-align:left">
        <li><b>📌 Multiple Input Methods</b> - Upload or capture images directly</li>
        <li><b>🔍 Advanced Analysis</b> - Comprehensive image understanding</li>
        <li><b>✍️ Custom Prompts</b> - Guide the AI with your specific requests</li>
        <li><b>📊 Detailed Reports</b> - Downloadable results in CSV format</li>
        <li><b>⚡ Fast Processing</b> - Quick turnaround with Gemini AI</li>
    </ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="text-align: center; margin-top: 3rem; color: #B0BEC5; font-size: 0.9rem;">
    <p>PIXTRACT - AI-Powered Image Analysis Tool</p>
</div>
""", unsafe_allow_html=True)
