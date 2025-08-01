import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import io
import base64
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from datetime import datetime
import streamlit.components.v1 as components

# Set page configuration
st.set_page_config(page_title="Lung Cancer Detection", layout="centered")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lung_cancer_model.h5")

model = load_model()

# Custom styles
st.markdown("""
    <style>
        body {
            background-color: #f9f9f9;
        }
        h1 {
            color: #0066cc;
        }
        .stButton>button {
            background-color: #0066cc;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1em;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Title and intro
st.markdown("<h1 style='text-align: center;'>🩺 LUNG CANCER DETECTION</h1>", unsafe_allow_html=True)
st.markdown("### Upload a Chest MRI image to detect possible signs of lung cancer.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a chest MRI image", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_lung_cancer(image):
    image = image.resize((224, 224)).convert("L")
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 224, 224, 1)
    prediction = model.predict(img_array)
    return prediction[0][0]

# PDF Report Generator
def generate_pdf(outcome, confidence):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    text = c.beginText(50, 780)
    text.setFont("Helvetica-Bold", 16)
    text.textLine("Lung Cancer Detection Report")

    text.setFont("Helvetica", 12)
    text.textLine("")
    text.textLine(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    text.textLine("")
    text.textLine("Analysis Result:")
    text.textLine(f"→ Diagnosis: {outcome}")
    text.textLine(f"→ Confidence Score: {confidence:.2f}%")

    c.drawText(text)
    c.showPage()
    c.save()
    pdf_data = buffer.getvalue()
    buffer.close()
    return pdf_data

# If image uploaded
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    filename = uploaded_file.name.lower()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    with col2:
        st.markdown("### 🔍 Result")
        if st.button("🧠 Analyze"):
            with st.spinner("Analyzing image with AI..."):
                time.sleep(1.5)
                result = predict_lung_cancer(image)
                confidence = round(float(result) * 100, 2)

                # Fix: Reversed logic — Assuming 0 = Cancer, 1 = No Cancer
                if result < 0.5:
                    outcome = "⚠️ Lung Cancer Detected"
                    color = "#e60000"
                    confidence = 100 - confidence
                else:
                    outcome = "✅ No Lung Cancer Detected"
                    color = "#009900"

                # Display result
                st.markdown(f"<h3 style='color: #333;'>Result: <span style='color: {color};'>{outcome}</span></h3>", unsafe_allow_html=True)
                st.progress(int(confidence))
                st.write(f"🧪 **Confidence Score:** {confidence:.2f}%")
                st.write(f"🔍 Raw Model Output: {result:.4f}")

                # Generate and offer PDF download
                pdf_data = generate_pdf(outcome, confidence)
                b64 = base64.b64encode(pdf_data).decode()

                button_html = f"""
                    <html>
                        <body>
                            <a download="lung_cancer_report.pdf" href="data:application/octet-stream;base64,{b64}" 
                               style="text-decoration: none;">
                                <button style="
                                    background-color: #0066cc;
                                    color: white;
                                    padding: 0.5em 1em;
                                    border: none;
                                    border-radius: 8px;
                                    font-weight: bold;
                                    cursor: pointer;
                                    font-size: 16px;
                                    margin-top: 10px;
                                ">📄 Download PDF Report</button>
                            </a>
                        </body>
                    </html>
                """
                components.html(button_html, height=80)

else:
    st.info("Please upload a Chest MRI image to begin analysis.")

# Footer
st.markdown("<hr><small>🧬 Developed by Priyanka kachhap</small>", unsafe_allow_html=True)
